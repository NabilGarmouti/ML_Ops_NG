# Problematique et dataset

## Contexte metier

Une compagnie d'assurance souhaite identifier les clients susceptibles d'etre interesses
par une assurance automobile. L'objectif est de prioriser les campagnes commerciales vers
les clients ayant la plus forte probabilite de repondre positivement, afin de reduire les
couts de prospection et d'ameliorer le taux de conversion.

## Probleme ML

Le projet est formule comme une classification binaire.

La cible est `Response` :

- `1` : le client est interesse par l'assurance automobile ;
- `0` : le client n'est pas interesse.

Le modele apprendra a partir de caracteristiques client et vehicule, par exemple l'age du
client, le fait qu'il possede deja une assurance, l'age du vehicule, les dommages deja subis
par le vehicule, la prime annuelle et le canal de vente.

Cette problematique est compatible avec la baseline du cours, car le dataset est tabulaire,
la cible est binaire et les variables melangent des colonnes numeriques et categorielles.

## Dataset choisi

Dataset Kaggle : Health Insurance Cross Sell Prediction

Lien : https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Raison du choix :

- domaine lie a l'industrie automobile via l'assurance vehicule ;
- cible binaire deja disponible (`Response`) ;
- colonnes numeriques et categorielles compatibles avec un pipeline scikit-learn ;
- cas metier facile a expliquer et utile pour une entreprise ;
- dataset suffisamment connu pour etre reutilisable dans un projet MLOps.

## Colonnes prevues pour la baseline

Cible :

- `Response`

Variables numeriques :

- `Age`
- `Region_Code`
- `Annual_Premium`
- `Policy_Sales_Channel`
- `Vintage`

Variables categorielles :

- `Gender`
- `Driving_License`
- `Previously_Insured`
- `Vehicle_Age`
- `Vehicle_Damage`

## Import du dataset

Le dataset sera importe plus tard dans le dossier `data/`, par exemple sous le nom :

```text
data/train.csv
```

Apres import, le fichier de configuration du pipeline devra pointer vers cette cible :

```python
DATA_PATH = ROOT / "data" / "train.csv"
TARGET = "Response"
NUMERIC_FEATURES = [
    "Age",
    "Region_Code",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Driving_License",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
]
```

## Avancement

### Etape 1 - Initialisation du projet

- creation du `Makefile` a la racine ;
- creation du `pyproject.toml` ;
- choix de la problematique metier ;
- choix du dataset Kaggle.

### Etape 2 - Configuration dataset

Structure ajoutee :

```text
src/
|-- __init__.py
|-- config.py
|-- data.py
`-- features.py
```

Role des fichiers :

- `config.py` centralise le chemin du dataset, la cible, les colonnes et les noms MLflow ;
- `data.py` lit le CSV brut et prepare le split train/test ;
- `features.py` nettoie les donnees, valide les colonnes attendues et construit le
  pre-processing scikit-learn avec imputation, standardisation et encodage one-hot.

La configuration supporte une surcouche par variables d'environnement pour les elements qui
peuvent changer selon l'environnement :

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT`
- `MODEL_NAME`
- `MODEL_STAGE`

Les stages prevus pour le cycle de vie du modele sont :

```python
MODEL_STAGES = ["integ", "prepod", "prod"]
```

Une fois le CSV Kaggle place dans `data/train.csv`, la validation se lance avec :

```bash
make data
```

Sur Windows, si `make` n'est pas installe, la commande equivalente est :

```bash
$env:PYTHONPATH = "src"
uv run python -m features
```

### Etape 3 - Baseline d'entrainement

Le fichier `src/train.py` entraine et compare cinq modeles :

- `logistic_regression`
- `decision_tree`
- `random_forest`
- `xgboost`
- `lightgbm`

Les metriques suivies sont :

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`

La commande d'entrainement est :

```bash
make train
```

Sur Windows, si `make` n'est pas installe :

```bash
$env:PYTHONPATH = "src"
uv run python -m train --sample-size 0 --selection-metric f1
```

Le choix du modele sauvegarde se fait avec `--selection-metric`. Par defaut, on utilise
`f1`, car le dataset est desequilibre et le meilleur `roc_auc` ne donne pas toujours le
meilleur compromis metier.

Sur le dataset complet, le benchmark actuel donne :

| modele | accuracy | precision | recall | f1 | roc_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 0.703 | 0.283 | 0.929 | 0.434 | 0.858 |
| decision_tree | 0.698 | 0.278 | 0.919 | 0.427 | 0.846 |
| logistic_regression | 0.641 | 0.251 | 0.974 | 0.399 | 0.839 |
| random_forest | 0.867 | 0.359 | 0.114 | 0.173 | 0.832 |
| xgboost | 0.877 | 0.000 | 0.000 | 0.000 | 0.856 |

Le modele retenu par defaut est donc `lightgbm`, car il obtient le meilleur `f1` sur
ce benchmark.

Artefacts generes localement :

```text
models/metrics.csv
models/model.joblib
models/confusion_matrix.png
```

### Etape 4 - Tracking MLflow

L'entrainement trace automatiquement les experiences avec MLflow.

Par defaut, le tracking est local :

```python
MLFLOW_TRACKING_URI = sqlite:///mlflow.db
```

Il peut etre surcharge via variable d'environnement :

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Chaque entrainement cree :

- un run parent `baseline_benchmark` ;
- un run enfant par modele entraine ;
- les parametres principaux ;
- les metriques `accuracy`, `precision`, `recall`, `f1`, `roc_auc` ;
- les artefacts `metrics.csv`, `confusion_matrix.png` et le meilleur modele.

Pour lancer l'interface MLflow :

```bash
make mlflow
```

Sur Windows, si `make` n'est pas installe :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

### Etape 5 - Modeles optimises

Le fichier `src/train_models.py` compare trois familles de modeles optimisees avec
`GridSearchCV` :

- `random_forest`
- `xgboost`
- `lightgbm`

Chaque famille est tracee dans MLflow avec :

- un run parent `compare-optimized-models` ;
- un run enfant par famille de modele ;
- les meilleurs hyperparametres trouves ;
- le score moyen de validation croisee ;
- les metriques de test `precision`, `recall`, `f1`, `roc_auc` ;
- un artefact d'explicabilite `shap_summary.png` ;
- le meilleur modele enregistre dans le Model Registry sous `cars-cross-sell-classifier`.

Le tracking MLflow est isole dans `src/tracking.py`. `train_models.py` se concentre sur
l'optimisation des modeles et appelle les fonctions de tracking dediees.

Commande rapide de test :

```bash
$env:PYTHONPATH = "src"
uv sync --extra dev --extra explain
uv run python -m train_models --cv 2 --scoring f1 --sample-size 1000
```

Commande complete :

```bash
$env:PYTHONPATH = "src"
uv sync --extra dev --extra explain
uv run python -m train_models --cv 5 --scoring f1 --sample-size 0
```

Les artefacts locaux generes sont :

```text
models/optimized_metrics.csv
models/optimized_confusion_matrix.png
models/model.joblib
```

### Etape 6 - API FastAPI

Le fichier `src/api.py` expose le modele sauvegarde dans `models/model.joblib`.

Le fichier `src/script.py` joue le role de client simple pour appeler l'endpoint
`POST /predict` avec un payload JSON.

Endpoints disponibles :

- `GET /health` : verifie que l'API repond ;
- `GET /model-info` : retourne les informations du modele servi ;
- `POST /predict` : retourne la prediction et la probabilite associee.

Lancement :

```bash
make api
```

Sur Windows, si `make` n'est pas installe :

```bash
$env:PYTHONPATH = "src"
uv run uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

Exemple de payload :

```json
{
  "Gender": "Male",
  "Age": 44,
  "Driving_License": 1,
  "Region_Code": 28.0,
  "Previously_Insured": 0,
  "Vehicle_Age": "> 2 Years",
  "Vehicle_Damage": "Yes",
  "Annual_Premium": 40454.0,
  "Policy_Sales_Channel": 26.0,
  "Vintage": 217
}
```

Appel du endpoint depuis le script client :

```bash
$env:PYTHONPATH = "src"
uv run python -m script
```

Le script peut aussi lire un payload depuis un fichier JSON :

```bash
$env:PYTHONPATH = "src"
uv run python -m script --payload payload.json
```

### Etape 7 - Optimisation avec Optuna

Le fichier `src/train_optuna.py` optimise les trois memes familles de modeles que
`train_models.py`, mais cette fois avec `Optuna` et un sampler `TPE` :

- `random_forest`
- `xgboost`
- `lightgbm`

L'objectif est de tester une vraie recherche bayesienne d'hyperparametres au lieu d'une
grille fixe.

Le tracking MLflow reste separe dans `src/tracking.py`. Pour cette etape, on loggue :

- un run parent `optuna-compare-models` ;
- un run enfant par famille de modele ;
- un run enfant supplementaire par `trial` Optuna ;
- les meilleurs hyperparametres de chaque famille ;
- les metriques `cv_score`, `precision`, `recall`, `f1`, `roc_auc` ;
- le `shap_summary.png` du meilleur pipeline de chaque famille ;
- le meilleur modele final dans le Model Registry.

Commande rapide de test :

```bash
$env:PYTHONPATH = "src"
uv run python -m train_optuna --n-trials 1 --cv 2 --scoring f1 --sample-size 300
```

Commande complete :

```bash
make train-optuna N_TRIALS=30 CV=5 SCORING=f1 SAMPLE_SIZE=0
```

Sur Windows, sans `make` :

```bash
$env:PYTHONPATH = "src"
uv run python -m train_optuna --n-trials 30 --cv 5 --scoring f1 --sample-size 0
```

Artefacts locaux generes :

```text
models/optuna_metrics.csv
models/optuna_confusion_matrix.png
models/model.joblib
```

### Etape 7 bis - Evaluation du modele

Le fichier `src/evaluate.py` reprend la logique du TP d'evaluation modele :

- recuperer la derniere version enregistree dans le Model Registry MLflow ;
- construire un jeu d'evaluation ;
- logger ce dataset dans MLflow pour la tracabilite ;
- executer `mlflow.models.evaluate` ;
- appliquer une porte qualite avec des seuils configurables.

Commande :

```bash
$env:PYTHONPATH = "src"
uv run python -m evaluate
```

Les seuils peuvent etre surcharges avec :

```bash
EVAL_ROC_AUC_MIN=0.65
EVAL_F1_MIN=0.20
```

### Etape 8 - Dockerisation

La partie Docker prepare le projet pour une execution plus stable et plus proche d'un
deploiement reel.

Fichiers ajoutes :

- `docker/Dockerfile.train` : image pour lancer l'entrainement
- `docker/Dockerfile.api` : image pour servir l'API FastAPI
- `docker/Dockerfile.frontend` : image pour le frontend Streamlit
- `docker-compose.yml` : stack locale `mlflow` + `train` + `api` + `frontend`

Commandes principales :

```bash
make docker-build
make docker-run
make docker-up
make docker-down
```

Details :

- `make docker-build` construit les images `cars-train`, `cars-api` et `cars-frontend`
- `make docker-run` lance l'entrainement one-shot via le service `train`
- `make docker-up` demarre MLflow sur `http://127.0.0.1:5000` et l'API sur
  `http://127.0.0.1:8000`, ainsi que le frontend sur `http://127.0.0.1:8501`
- `make docker-down` arrete la stack

La stack Docker continue d'utiliser :

- un volume `models_data` partage entre `train` et `api`
- un volume `mlflow_data` pour conserver MLflow
- le dossier `data/` monte dans le conteneur `train`
- `API_URL=http://api:8000` dans le frontend via le DNS interne Docker Compose

Exemple de sequence :

```bash
make docker-build
make docker-run
make docker-up
```

Puis ouvrir :

```text
http://127.0.0.1:5000
http://127.0.0.1:8000/docs
http://127.0.0.1:8501
```

### Etape 9 - Workflow Docker reproductible

Le Makefile contient un workflow complet pour repartir d'un environnement Docker propre,
reconstruire les images, re-entrainer le modele et relancer les services.

Commande de demonstration locale :

```bash
make deploy-local SAMPLE_SIZE=50000 CV=2 N_TRIALS=5
```

Cette commande execute :

- demarrage de MLflow ;
- entrainement `train_models` avec GridSearchCV ;
- entrainement `train_optuna` avec Optuna ;
- generation de `models/model.joblib` ;
- demarrage de `mlflow`, `api`, `frontend` et `airflow`.

Pour un test plus rapide, reduire `SAMPLE_SIZE` :

```bash
make deploy-local SAMPLE_SIZE=5000 CV=2 N_TRIALS=5
```

Services exposes localement :

```text
Frontend Streamlit : http://127.0.0.1:8501
API Swagger        : http://127.0.0.1:8000/docs
MLflow             : http://127.0.0.1:5000
Airflow            : http://127.0.0.1:8080
```

### Etape 10 - Orchestration Airflow

Airflow orchestre deux DAGs separes :

- `cars_training_pipeline` : validation dataset, GridSearchCV, Optuna, evaluation ;
- `cars_predict_pipeline` : verification de l'API puis envoi de payloads de test via
  `src/script.py`.

Schedules :

- entrainement : toutes les 4 heures, cron `0 */4 * * *` ;
- prediction : toutes les 2 heures, cron `0 */2 * * *`.

Les logs du DAG de prediction permettent de retrouver les payloads envoyes et les reponses
retournees par l'API.

Airflow est lance via Docker Compose avec :

```bash
make airflow
```

Recuperation du mot de passe Airflow :

```bash
make airflow-password
```

Identifiant par defaut :

```text
user: admin
password: affiche par make airflow-password
```
