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
data/dataset.csv
```

Apres import, le fichier de configuration du pipeline devra pointer vers cette cible :

```python
DATA_PATH = ROOT / "data" / "dataset.csv"
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
