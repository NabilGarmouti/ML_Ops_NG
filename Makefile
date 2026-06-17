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
AIRFLOW_PORT  ?= 8080
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
SCORING       ?= f1
N_TRIALS      ?= 30
SAMPLE_SIZE   ?= 0
SELECTION_METRIC ?= f1

ifeq ($(OS),Windows_NT)
YELLOW :=
GREEN  :=
RED    :=
CYAN   :=
RESET  :=
else
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')
endif

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna evaluate mlflow api predict-api frontend \
        docker-build docker-run docker-up docker-down share \
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

evaluate: ## Evalue la derniere version enregistree du modele MLflow
	$(PYTHON) -m evaluate

mlflow: ## Demarre le serveur MLflow (docker compose)
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $(MLFLOW_PORT)

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn api:app --reload --host $(API_HOST) --port $(API_PORT)

predict-api: ## Envoie un payload d'exemple au endpoint /predict
	$(PYTHON) -m script

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

docker-build: ## Construit les images Docker du projet
	docker build -f docker/Dockerfile.train -t cars-train .
	docker build -f docker/Dockerfile.api -t cars-api .
	docker build -f docker/Dockerfile.frontend -t cars-frontend .

docker-run: ## Lance l'entrainement en conteneur via docker compose
	docker compose -f docker-compose.yml --profile train run --rm train

docker-up: ## Demarre la stack Docker (mlflow + api + frontend)
	docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrete et supprime la stack Docker
	docker compose -f docker-compose.yml down

ifeq ($(OS),Windows_NT)
share: ## Affiche les URLs LAN a partager depuis Windows
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/share_urls.ps1 -ApiPort $(API_PORT) -FrontendPort $(FRONTEND_PORT) -MlflowPort $(MLFLOW_PORT) -AirflowPort $(AIRFLOW_PORT)
else
share: ## Affiche les URLs LAN a partager depuis macOS/Linux
	@IP=$$( \
		IFACE=$$(route -n get default 2>/dev/null | awk '/interface:/{print $$2}'); \
		ipconfig getifaddr "$$IFACE" 2>/dev/null \
		|| ifconfig 2>/dev/null | awk '/inet /{print $$2}' | grep -v '^127\.' | head -1 \
		|| hostname -I 2>/dev/null | awk '{print $$1}' \
	); \
	if [ -z "$$IP" ]; then echo "$(RED)[ERREUR] IP LAN introuvable$(RESET)"; exit 1; fi; \
	echo "$(GREEN)IP LAN de la machine : $$IP$(RESET)"; \
	echo ""; \
	echo "$(CYAN)URLs a partager (autres machines du meme reseau Wi-Fi/LAN) :$(RESET)"; \
	for entry in "API (docs):$(API_PORT):/docs" "Frontend:$(FRONTEND_PORT):" "MLflow:$(MLFLOW_PORT):" "Airflow:$(AIRFLOW_PORT):"; do \
		name=$$(echo "$$entry" | cut -d: -f1); \
		port=$$(echo "$$entry" | cut -d: -f2); \
		path=$$(echo "$$entry" | cut -d: -f3); \
		if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:$$port -sTCP:LISTEN -t >/dev/null 2>&1; then \
			state="$(GREEN)actif$(RESET)"; \
		else \
			state="$(YELLOW)hors ligne$(RESET)"; \
		fi; \
		printf " %-11s http://%s:%s%s [%b]\n" "$$name" "$$IP" "$$port" "$$path" "$$state"; \
	done; \
	echo ""; \
	echo "$(YELLOW)Les services Docker sont exposes en 0.0.0.0 via docker compose."; \
	echo "Si une autre machine ne peut pas se connecter, verifier le pare-feu ou l'isolation client du routeur.$(RESET)"
endif

lint: ## Verifie le style (ruff)
	$(RUN) ruff check src tests frontend

format: ## Formate le code (ruff)
	$(RUN) ruff format src tests frontend

type: ## Verifie les types (mypy)
	$(PYTHON) -m compileall -q src

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
