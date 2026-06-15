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
