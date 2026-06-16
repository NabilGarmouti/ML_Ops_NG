# ==============================================================================
# Cars - Projet MLOps automobile (squelette)
# ==============================================================================

SHELL         := /bin/sh
PYTHON        := uv run python
RUN           := uv run
VENV_DIR      := .venv
PYTHONPATH    ?= src
export PYTHONPATH
API_HOST      ?= 127.0.0.1
API_PORT      ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT   := 5000
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
SCORING       ?= f1
N_TRIALS      ?= 30
SAMPLE_SIZE   ?= 0
SELECTION_METRIC ?= f1

YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"

data: ## Valide le CSV Kaggle place dans data/dataset.csv
	$(PYTHON) -m features

train: ## Entraine la baseline -> models/model.joblib (C=.. MAX_ITER=..)
	$(PYTHON) -m train --sample-size $(SAMPLE_SIZE) --selection-metric $(SELECTION_METRIC) --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) + SHAP (CV=.. SCORING=..)
	$(PYTHON) -m train_models --cv $(CV) --scoring $(SCORING) --sample-size $(SAMPLE_SIZE)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	$(PYTHON) -m train_optuna --n-trials $(N_TRIALS) --cv $(CV) --scoring $(SCORING) --sample-size $(SAMPLE_SIZE)

mlflow: ## Demarre le serveur MLflow (docker compose)
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $(MLFLOW_PORT)

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	# TODO (S14bis) : $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

docker-build: ## Construit l'image d'entrainement
	# TODO (S8) : docker build -f docker/Dockerfile.train -t cars-train .

docker-run: ## Lance l'entrainement en conteneur
	# TODO (S8) : docker run --rm -v "$(CURDIR)/models:/app/models" cars-train

docker-up: ## Demarre la stack (mlflow, api, frontend)
	# TODO (S14) : docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	# TODO (S14) : docker compose -f docker-compose.yml down

lint: ## Verifie le style (ruff)
	$(RUN) ruff check src

format: ## Formate le code (ruff)
	$(RUN) ruff format src

type: ## Verifie les types (mypy)
	$(RUN) mypy src

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
