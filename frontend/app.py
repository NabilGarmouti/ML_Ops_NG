"""Frontend Streamlit pour presenter et tester le projet Cars Cross-Sell."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
AUTHOR_NAME = "Nabil Garmouti"
REPO_URL = "https://github.com/NabilGarmouti/ML_Ops_NG"
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

METRIC_FILES = {
    "Baseline - 5 modeles": MODELS_DIR / "metrics.csv",
    "GridSearchCV - 3 modeles optimises": MODELS_DIR / "optimized_metrics.csv",
    "Optuna - recherche bayesienne": MODELS_DIR / "optuna_metrics.csv",
}

FEATURE_EXAMPLE = {
    "Gender": "Male",
    "Age": 44,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Vehicle_Age": "> 2 Years",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 40454.0,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 217,
}


st.set_page_config(page_title="Cars Cross-Sell", layout="wide")


def render_theme() -> None:
    """Apply a compact visual theme on top of Streamlit defaults."""
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2.4rem;
            max-width: 1220px;
          }

          div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(28, 34, 48, 0.96), rgba(17, 24, 39, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            padding: 14px 16px;
          }

          .hero {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 20px;
            background:
              linear-gradient(135deg, rgba(14, 165, 233, 0.25), rgba(34, 197, 94, 0.13)),
              linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.98));
          }

          .hero-top {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-start;
            flex-wrap: wrap;
          }

          .hero h1 {
            font-size: 2.35rem;
            margin: 0 0 8px 0;
            letter-spacing: 0;
          }

          .hero p {
            max-width: 820px;
            color: rgba(226, 232, 240, 0.9);
            font-size: 1.02rem;
            margin-bottom: 16px;
          }

          .hero-author {
            color: rgba(186, 230, 253, 0.96);
            font-weight: 650;
            margin-bottom: 12px;
          }

          .badges {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
          }

          .badge {
            border: 1px solid rgba(226, 232, 240, 0.24);
            border-radius: 999px;
            padding: 5px 11px;
            background: rgba(15, 23, 42, 0.5);
            color: rgba(248, 250, 252, 0.94);
            font-size: 0.86rem;
          }

          .hero-link {
            display: inline-block;
            border: 1px solid rgba(125, 211, 252, 0.48);
            border-radius: 8px;
            padding: 10px 14px;
            background: rgba(8, 47, 73, 0.58);
            color: #f8fafc !important;
            text-decoration: none !important;
            font-weight: 700;
            white-space: nowrap;
          }

          .service-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(150px, 1fr));
            gap: 12px;
            margin: 8px 0 18px 0;
          }

          .service-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 14px 15px;
            background: rgba(15, 23, 42, 0.54);
            color: #f8fafc !important;
            text-decoration: none !important;
            min-height: 88px;
          }

          .service-card:hover {
            border-color: rgba(125, 211, 252, 0.7);
            background: rgba(15, 23, 42, 0.82);
          }

          .service-label {
            font-weight: 750;
            margin-bottom: 7px;
          }

          .service-url {
            color: rgba(203, 213, 225, 0.78);
            font-size: 0.82rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
          }

          .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            margin-right: 8px;
          }

          .section-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            padding: 18px 20px;
            background: rgba(15, 23, 42, 0.36);
            margin-bottom: 16px;
          }

          .section-card h3 {
            margin-top: 0;
            margin-bottom: 8px;
          }

          .muted {
            color: rgba(226, 232, 240, 0.78);
          }

          div[data-testid="stTabs"] button {
            font-weight: 650;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the dashboard header."""
    st.markdown(
        """
        <div class="hero">
          <div class="hero-top">
            <div>
              <h1>Cars Cross-Sell</h1>
              <div class="hero-author">Projet realise par Nabil Garmouti</div>
              <p>
                Projet MLOps de classification binaire pour prioriser les clients les plus
                susceptibles de souscrire une assurance automobile.
              </p>
            </div>
            <a class="hero-link" href="https://github.com/NabilGarmouti/ML_Ops_NG"
               target="_blank" rel="noopener noreferrer">
              Repo GitHub
            </a>
          </div>
          <div class="badges">
            <span class="badge">FastAPI</span>
            <span class="badge">Streamlit</span>
            <span class="badge">MLflow</span>
            <span class="badge">Airflow</span>
            <span class="badge">Docker Compose</span>
            <span class="badge">GridSearchCV</span>
            <span class="badge">Optuna</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    """Render a small information card."""
    st.markdown(
        f"""
        <div class="section-card">
          <h3>{title}</h3>
          <div class="muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_metrics() -> dict[str, pd.DataFrame]:
    """Load available model metrics generated by training scripts."""
    metrics: dict[str, pd.DataFrame] = {}
    for label, path in METRIC_FILES.items():
        if path.exists():
            frame = pd.read_csv(path)
            metrics[label] = frame
    return metrics


def get_api_json(path: str) -> tuple[bool, dict | str]:
    """Call a JSON API endpoint and return a display-friendly result."""
    try:
        response = httpx.get(f"{api_url}{path}", timeout=5.0)
        response.raise_for_status()
        return True, response.json()
    except httpx.HTTPError as exc:
        return False, str(exc)


def format_metric(value: float) -> str:
    """Format a metric for compact display."""
    return f"{value:.3f}"


def best_row(frame: pd.DataFrame, metric: str = "f1") -> pd.Series:
    """Return the best row for a metric, falling back to the first row."""
    if metric in frame.columns:
        return frame.sort_values(metric, ascending=False).iloc[0]
    return frame.iloc[0]


def render_public_links() -> None:
    """Render service links using the hostname used by the browser."""
    components.html(
        """
        <style>
          body { margin: 0; }
          #public-links {
            display: grid;
            grid-template-columns: repeat(5, minmax(150px, 1fr));
            gap: 12px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          }
          .service-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 14px 15px;
            background: rgba(15, 23, 42, 0.62);
            color: #f8fafc;
            text-decoration: none;
            min-height: 86px;
            box-sizing: border-box;
          }
          .service-card:hover {
            border-color: rgba(125, 211, 252, 0.82);
            background: rgba(15, 23, 42, 0.9);
          }
          .service-label {
            font-size: 15px;
            font-weight: 760;
            margin-bottom: 8px;
          }
          .service-url {
            color: rgba(203, 213, 225, 0.78);
            font-size: 12px;
            line-height: 1.25;
            overflow-wrap: anywhere;
          }
          .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            margin-right: 8px;
          }
          @media (max-width: 900px) {
            #public-links { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
          }
        </style>
        <div id="public-links"></div>
        <script>
          const services = [
            {label: "Repo GitHub", href: "https://github.com/NabilGarmouti/ML_Ops_NG"},
            {label: "MLflow", port: "5000", path: ""},
            {label: "Swagger API", port: "8000", path: "/docs"},
            {label: "Frontend", port: "8501", path: ""},
            {label: "Airflow", port: "8080", path: ""}
          ];

          function currentHost() {
            try {
              return window.parent.location.hostname || window.location.hostname;
            } catch (error) {
              return window.location.hostname || "127.0.0.1";
            }
          }

          function currentProtocol() {
            try {
              return window.parent.location.protocol || window.location.protocol || "http:";
            } catch (error) {
              return window.location.protocol || "http:";
            }
          }

          const host = currentHost();
          const protocol = currentProtocol();
          const container = document.getElementById("public-links");

          container.innerHTML = services.map((service) => {
            const href = service.href || `${protocol}//${host}:${service.port}${service.path}`;
            const shownUrl = service.href || `${host}:${service.port}${service.path}`;
            return `
              <a href="${href}" target="_blank" rel="noopener noreferrer"
                 class="service-card">
                <div class="service-label"><span class="status-dot"></span>${service.label}</div>
                <div class="service-url">${shownUrl}</div>
              </a>`;
          }).join("");
        </script>
        """,
        height=118,
    )


render_theme()
render_hero()

with st.sidebar:
    st.caption(f"Projet realise par {AUTHOR_NAME}")
    st.header("Pilotage")
    st.link_button("Repo GitHub", REPO_URL, use_container_width=True)
    api_url = st.text_input("URL de l'API", value=API_URL)
    ok, health = get_api_json("/health")
    if ok:
        st.success("API disponible")
    else:
        st.error("API indisponible")
        st.caption(str(health))
    st.caption("Les predictions sont servies par le modele monte dans le container API.")

home_tab, predict_tab, models_tab, architecture_tab = st.tabs(
    ["Accueil", "Prediction", "Modeles", "API, Docker & Airflow"]
)

with home_tab:
    st.subheader("Acces rapides")
    render_public_links()

    st.subheader("Problematique metier")
    render_card(
        "Objectif",
        "Identifier les clients les plus susceptibles d'etre interesses par une "
        "assurance automobile afin de concentrer les campagnes commerciales sur les "
        "profils les plus prometteurs.",
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Type de probleme", "Classification")
    col_b.metric("Cible", "Response")
    col_c.metric("Taux positif", "12.3%")
    col_d.metric("Cas metier", "Cross-sell")

    home_metrics = load_metrics()
    if home_metrics:
        latest_label, latest_frame = list(home_metrics.items())[-1]
        latest_best = best_row(latest_frame)
        st.subheader("Etat du modele")
        model_a, model_b, model_c, model_d = st.columns(4)
        model_a.metric("Derniere strategie", latest_label.split(" - ")[0])
        model_b.metric("Meilleur modele", str(latest_best["model"]))
        model_c.metric("F1", format_metric(float(latest_best["f1"])))
        model_d.metric("ROC AUC", format_metric(float(latest_best["roc_auc"])))

    st.subheader("Dataset")
    render_card(
        "Health Insurance Cross Sell Prediction",
        "Dataset Kaggle tabulaire. Chaque ligne represente un client avec des "
        "informations client, vehicule, assurance et canal commercial.",
    )

    var_a, var_b, var_c = st.columns(3)
    with var_a:
        render_card("Client", "`Gender`, `Age`, `Region_Code`, `Vintage`")
    with var_b:
        render_card("Assurance", "`Driving_License`, `Previously_Insured`, `Annual_Premium`")
    with var_c:
        render_card("Vehicule & canal", "`Vehicle_Age`, `Vehicle_Damage`, `Policy_Sales_Channel`")

    st.subheader("Ce qui a ete construit")
    steps = pd.DataFrame(
        [
            ["1", "Preparation", "Lecture CSV, nettoyage, split, preprocessing"],
            ["2", "Baseline", "Comparaison de 5 modeles avec MLflow"],
            ["3", "Optimisation", "GridSearchCV sur Random Forest, XGBoost, LightGBM"],
            ["4", "Optuna", "Recherche bayesienne et tracking des trials"],
            ["5", "Evaluation", "Courbes, matrice de confusion, ROC AUC, lift"],
            ["6", "Serving", "API FastAPI, client de prediction, frontend Streamlit"],
            ["7", "Orchestration", "Deux DAGs Airflow : entrainement et prediction"],
            ["8", "Industrialisation", "Docker Compose, CI GitHub Actions, CD GHCR"],
        ],
        columns=["Etape", "Bloc", "Resultat"],
    )
    st.dataframe(steps, hide_index=True, use_container_width=True)

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        with st.form("predict_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=FEATURE_EXAMPLE["Age"])
            driving_license = st.selectbox("Driving_License", [0, 1], index=1)
            region_code = st.number_input(
                "Region_Code", min_value=0.0, value=FEATURE_EXAMPLE["Region_Code"]
            )
            previously_insured = st.selectbox("Previously_Insured", [0, 1], index=0)
            vehicle_age = st.selectbox(
                "Vehicle_Age", ["< 1 Year", "1-2 Year", "> 2 Years"], index=2
            )
            vehicle_damage = st.selectbox("Vehicle_Damage", ["No", "Yes"], index=1)
            annual_premium = st.number_input(
                "Annual_Premium", min_value=0.0, value=FEATURE_EXAMPLE["Annual_Premium"]
            )
            sales_channel = st.number_input(
                "Policy_Sales_Channel",
                min_value=0.0,
                value=FEATURE_EXAMPLE["Policy_Sales_Channel"],
            )
            vintage = st.number_input("Vintage", min_value=0, value=FEATURE_EXAMPLE["Vintage"])
            submitted = st.form_submit_button("Predire")

    with col_result:
        if submitted:
            payload = {
                "Gender": gender,
                "Age": int(age),
                "Driving_License": int(driving_license),
                "Region_Code": float(region_code),
                "Previously_Insured": int(previously_insured),
                "Vehicle_Age": vehicle_age,
                "Vehicle_Damage": vehicle_damage,
                "Annual_Premium": float(annual_premium),
                "Policy_Sales_Channel": float(sales_channel),
                "Vintage": int(vintage),
            }

            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Appel a l'API impossible : {exc}")
            else:
                prediction = int(result["prediction"])
                probability = float(result["probability"])
                label = "Interesse" if prediction == 1 else "Non interesse"

                st.metric("Decision modele", label)
                st.metric("Probabilite d'interet", f"{probability:.2%}")
                st.progress(min(max(probability, 0.0), 1.0))

                if probability >= 0.5:
                    st.success("Le client peut etre priorise dans une campagne commerciale.")
                else:
                    st.info("Le client n'est pas prioritaire avec le seuil actuel de 0.5.")

                st.json(payload)
        else:
            st.info("Renseigne un profil client puis lance la prediction.")

with models_tab:
    st.subheader("Historique des experimentations")
    metrics = load_metrics()

    if not metrics:
        st.warning("Aucun fichier de metriques trouve dans le dossier models/.")
    else:
        summary_rows = []
        for version, frame in metrics.items():
            row = best_row(frame)
            summary_rows.append(
                {
                    "version": version,
                    "meilleur_modele": row["model"],
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1": row.get("f1"),
                    "roc_auc": row.get("roc_auc"),
                }
            )

        summary = pd.DataFrame(summary_rows)
        best_overall = best_row(summary.rename(columns={"meilleur_modele": "model"}))
        card_a, card_b, card_c = st.columns(3)
        card_a.metric("Meilleure version", str(best_overall["version"]).split(" - ")[0])
        card_b.metric("Modele servi", str(best_overall["model"]))
        card_c.metric("F1 observe", format_metric(float(best_overall["f1"])))

        st.dataframe(
            summary.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_data = summary.set_index("version")[["f1", "roc_auc"]]
        st.bar_chart(chart_data)

        selected_version = st.selectbox("Details d'une experimentation", list(metrics))
        frame = metrics[selected_version]
        st.dataframe(frame, use_container_width=True, hide_index=True)

    st.subheader("Modele servi actuellement")
    ok, model_info = get_api_json("/model-info")
    if ok:
        st.json(model_info)
    else:
        st.error(model_info)

    st.info(
        "Les versions affichees ici representent les grandes etapes du projet : baseline, "
        "modeles optimises par GridSearchCV, puis optimisation Optuna. MLflow conserve le "
        "detail complet des runs, parametres, artefacts et versions enregistrees."
    )

with architecture_tab:
    st.subheader("Architecture de demonstration")
    st.markdown(
        """
        Services lances par Docker Compose :

        - `mlflow` : suivi des experiences et artefacts
        - `api` : service FastAPI qui expose `/health`, `/model-info` et `/predict`
        - `frontend` : interface Streamlit pour expliquer le projet et tester le modele
        - `airflow` : orchestration avec deux DAGs separes, un pour l'entrainement
          et un pour les predictions de test

        Le modele servi est lu depuis `models/model.joblib`, monte dans le container API.
        """
    )

    render_public_links()
    st.caption(
        "Les liens utilisent automatiquement l'adresse depuis laquelle le frontend est ouvert. "
        "Sur le reseau local, ils pointeront donc vers l'IP LAN de ta machine."
    )

    st.info(
        "Airflow orchestre deux DAGs distincts : `cars_training_pipeline` pour "
        "l'entrainement et `cars_predict_pipeline` pour les predictions de test."
    )
