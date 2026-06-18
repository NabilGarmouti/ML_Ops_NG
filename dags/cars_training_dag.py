"""DAG Airflow dedie au re-entrainement du modele Cars Cross-Sell."""

from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"


def project_command(command: str) -> str:
    """Build a shell command executed from the mounted project directory."""
    return f"cd {PROJECT_DIR} && {command}"


with DAG(
    dag_id="cars_training_pipeline",
    description="Valide les donnees, entraine les modeles optimises et evalue le modele final.",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 */4 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["cars", "training", "mlops"],
) as dag:
    sample_size = Variable.get("CARS_SAMPLE_SIZE", default_var="5000")
    cv = Variable.get("CARS_CV", default_var="2")
    n_trials = Variable.get("CARS_N_TRIALS", default_var="5")
    scoring = Variable.get("CARS_SCORING", default_var="f1")

    validate_dataset = BashOperator(
        task_id="validate_dataset",
        bash_command=project_command("python -m features"),
    )

    train_gridsearch = BashOperator(
        task_id="train_gridsearch_models",
        bash_command=project_command(
            "python -m train_models "
            f"--cv {cv} "
            f"--scoring {scoring} "
            f"--sample-size {sample_size}"
        ),
    )

    train_optuna = BashOperator(
        task_id="train_optuna_models",
        bash_command=project_command(
            "python -m train_optuna "
            f"--n-trials {n_trials} "
            f"--cv {cv} "
            f"--scoring {scoring} "
            f"--sample-size {sample_size}"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_registered_model",
        bash_command=project_command("python -m evaluate --no-validate"),
    )

    validate_dataset >> train_gridsearch >> train_optuna >> evaluate_model
