"""DAG Airflow dedie au test de prediction via l'API Cars Cross-Sell."""

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
    dag_id="cars_predict_pipeline",
    description="Verifie l'API et lance des predictions de test via le client projet.",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["cars", "predict", "api"],
) as dag:
    api_url = Variable.get("CARS_API_URL", default_var="http://api:8000")

    check_api_health = BashOperator(
        task_id="check_api_health",
        bash_command=f"curl --fail {api_url}/health",
    )

    run_prediction_client = BashOperator(
        task_id="run_prediction_client",
        bash_command=project_command(
            f"API_URL={api_url} python -m script --url {api_url}"
        ),
    )

    check_api_health >> run_prediction_client
