"""Interactive dashboard for the clinical trial outcome project.

Run:
    python dashboard.py

For model comparison and explainability, run these notebooks first:
    Random_Forest_Model.ipynb
    XGBoost_Model.ipynb
    Explainability_Analysis.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate


JOINED_PATH = Path("data/joined_dataset.parquet")
MODEL_READY_PATH = Path("data/model_ready.parquet")
METRICS_PATH = Path("artifacts/model_metrics.json")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")
EXPLAIN_DIR = Path("artifacts/explainability")
MODEL_DIR = Path("artifacts/models")

LABEL_NAMES = {0: "COMPLETED", 1: "TERMINATED", 2: "WITHDRAWN"}
OUTCOME_ORDER = ["COMPLETED", "TERMINATED", "WITHDRAWN"]
OUTCOME_COLORS = {
    "COMPLETED": "#2E7D32",
    "TERMINATED": "#C62828",
    "WITHDRAWN": "#1F6FEB",
}
METRIC_COLORS = {
    "Macro F1": "#047D95",
    "Accuracy": "#1F6FEB",
    "Weighted F1": "#5BA9D6",
}
FEATURE_GROUP_COLORS = {
    "Trial design": "#047D95",
    "Sponsor": "#1F6FEB",
    "OpenFDA signal": "#5BA9D6",
    "Eligibility": "#63C7B2",
    "Other": "#8AB6D6",
}
LOGISTIC_BASELINE_METRICS = {
    "name": "Logistic Regression Baseline",
    "slug": "logistic_regression",
    "accuracy": 0.773,
    "macro_f1": 0.8052,
    "weighted_f1": 0.773,
    "classification_report": {
        "COMPLETED": {"precision": 0.831, "recall": 0.682, "f1-score": 0.749, "support": 6144},
        "TERMINATED": {"precision": 0.664, "recall": 0.779, "f1-score": 0.717, "support": 4921},
        "WITHDRAWN": {"precision": 0.904, "recall": 1.000, "f1-score": 0.949, "support": 2354},
        "macro avg": {"precision": 0.800, "recall": 0.820, "f1-score": 0.805, "support": 13419},
        "weighted avg": {"precision": 0.783, "recall": 0.773, "f1-score": 0.773, "support": 13419},
    },
    "confusion_matrix": None,
    "test_rows": 13419,
    "num_features": 147,
    "split_year": 2019,
    "source": "Baseline.ipynb",
}


def load_joined_dataset() -> pl.DataFrame:
    if not JOINED_PATH.exists():
        raise FileNotFoundError(f"Missing {JOINED_PATH}. Run the join and preprocessing notebook first.")
    return pl.read_parquet(JOINED_PATH)


def load_model_dataset() -> pl.DataFrame | None:
    if not MODEL_READY_PATH.exists():
        return None
    return pl.read_parquet(MODEL_READY_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        payload = {"models": [], "label_names": LABEL_NAMES}
    else:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    models = payload.get("models", [])
    if not any(model.get("slug") == "logistic_regression" for model in models):
        models = [LOGISTIC_BASELINE_METRICS] + models
    payload["models"] = models
    return payload


def load_feature_columns() -> list[str]:
    if not FEATURE_COLUMNS_PATH.exists():
        return []
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def load_importance(slug: str, kind: str) -> pd.DataFrame:
    path = EXPLAIN_DIR / f"{slug}_{kind}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.read_csv(path)
    if "importance_mean" in df.columns:
        df = df.rename(columns={"importance_mean": "importance"})
    if "mean_abs_shap" in df.columns:
        df = df.rename(columns={"mean_abs_shap": "importance"})
    return df


def card(children, class_name: str = "card"):
    return html.Div(children, className=class_name)


def metric_card(label: str, value: str, note: str = ""):
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(note, className="metric-note"),
        ],
        className="metric-card",
    )


def make_table(df: pd.DataFrame, page_size: int = 10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Inter, Segoe UI, sans-serif",
            "fontSize": "13px",
            "padding": "8px",
            "textAlign": "left",
            "minWidth": "90px",
            "maxWidth": "260px",
            "whiteSpace": "normal",
        },
        style_header={"fontWeight": "700", "backgroundColor": "#F4F6F8"},
    )


def filter_joined(df: pl.DataFrame, outcomes, sponsors, fda_filter, year_range) -> pl.DataFrame:
    out = df
    if outcomes:
        out = out.filter(pl.col("overall_status").is_in(outcomes))
    if sponsors:
        out = out.filter(pl.col("sponsor_class").is_in(sponsors))
    if fda_filter == "matched":
        out = out.filter(pl.col("has_fda_record") == 1)
    elif fda_filter == "unmatched":
        out = out.filter(pl.col("has_fda_record") == 0)
    if year_range:
        out = out.filter(pl.col("start_year").is_between(year_range[0], year_range[1]))
    return out


def outcome_distribution_figure(df: pl.DataFrame):
    counts = (
        df.group_by("overall_status")
        .agg(pl.len().alias("trials"))
        .sort("overall_status")
        .to_pandas()
    )
    fig = px.bar(
        counts,
        x="overall_status",
        y="trials",
        color="overall_status",
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"overall_status": OUTCOME_ORDER},
        title="Trial Outcomes",
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig


def sponsor_outcome_figure(df: pl.DataFrame):
    grouped = (
        df.group_by(["sponsor_class", "overall_status"])
        .agg(pl.len().alias("trials"))
        .to_pandas()
    )
    fig = px.bar(
        grouped,
        x="sponsor_class",
        y="trials",
        color="overall_status",
        barmode="group",
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"overall_status": OUTCOME_ORDER},
        title="Outcome by Sponsor Class",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=80), height=380, xaxis_tickangle=-20)
    return fig


def fda_match_figure(df: pl.DataFrame):
    grouped = (
        df.group_by("overall_status")
        .agg(
            pl.len().alias("total"),
            pl.col("has_fda_record").cast(pl.Int32).sum().alias("matched"),
        )
        .with_columns((pl.col("matched") / pl.col("total") * 100).alias("pct_matched"))
        .to_pandas()
    )
    fig = px.bar(
        grouped,
        x="overall_status",
        y="pct_matched",
        color="overall_status",
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"overall_status": OUTCOME_ORDER},
        title="OpenFDA Match Rate by Outcome",
        labels={"pct_matched": "% matched"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20), height=360)
    fig.update_yaxes(range=[0, 100])
    return fig


def enrollment_figure(df: pl.DataFrame):
    plot_df = (
        df.select(["overall_status", "log_enrollment", "has_fda_record"])
        .drop_nulls()
        .sample(min(8000, df.height), seed=42)
        .to_pandas()
    )
    plot_df["FDA record"] = np.where(plot_df["has_fda_record"] == 1, "Matched", "Unmatched")
    fig = px.violin(
        plot_df,
        x="overall_status",
        y="log_enrollment",
        color="overall_status",
        box=True,
        points=False,
        color_discrete_map=OUTCOME_COLORS,
        category_orders={"overall_status": OUTCOME_ORDER},
        title="Enrollment Distribution by Outcome",
        labels={"log_enrollment": "log(1 + enrollment)"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig


def model_metric_figure(metrics: dict):
    rows = [
        {
            "model": model["name"],
            "Macro F1": model["macro_f1"],
            "Accuracy": model["accuracy"],
            "Weighted F1": model["weighted_f1"],
        }
        for model in metrics.get("models", [])
    ]
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="Run the model notebooks to generate model artifacts.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    metric_df = pd.DataFrame(rows).melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        metric_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Model Comparison",
        color_discrete_map=METRIC_COLORS,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=380)
    fig.update_yaxes(range=[0, 1])
    return fig


def confusion_matrix_figure(metrics: dict, slug: str):
    model = next((m for m in metrics.get("models", []) if m["slug"] == slug), None)
    if model is None:
        return go.Figure()
    if model.get("confusion_matrix") is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Baseline metrics are included from Baseline.ipynb. Re-run that notebook to export a confusion matrix artifact.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    cm = np.array(model["confusion_matrix"])
    fig = px.imshow(
        cm,
        x=OUTCOME_ORDER,
        y=OUTCOME_ORDER,
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"{model['name']} Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual", "color": "Trials"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig


def report_table(metrics: dict, slug: str) -> pd.DataFrame:
    model = next((m for m in metrics.get("models", []) if m["slug"] == slug), None)
    if model is None:
        return pd.DataFrame({"message": ["Run the model notebooks to generate reports."]})

    rows = []
    report = model["classification_report"]
    for label in OUTCOME_ORDER + ["macro avg", "weighted avg"]:
        item = report.get(label)
        if item:
            rows.append(
                {
                    "class": label,
                    "precision": round(item["precision"], 3),
                    "recall": round(item["recall"], 3),
                    "f1-score": round(item["f1-score"], 3),
                    "support": int(item["support"]),
                }
            )
    return pd.DataFrame(rows)


def feature_group(feature: str) -> str:
    if feature.startswith(("phases_", "intervention_model_", "primary_purpose_", "masking_", "num_", "log_enrollment", "trial_duration", "enrollment")):
        return "Trial design"
    if feature.startswith(("sponsor_class_", "num_collaborators")):
        return "Sponsor"
    if feature.startswith(("application_type_", "marketing_status_", "therapeutic_", "mechanism_", "route_", "dosage_", "approval", "years_since", "priority", "has_fda")):
        return "OpenFDA signal"
    if feature.startswith(("min_age", "max_age", "age_range", "sex_", "healthy")):
        return "Eligibility"
    return "Other"


def importance_figure(slug: str, kind: str, top_n: int):
    df = load_importance(slug, kind)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No explainability artifact found for this model.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=420)
        return fig

    plot_df = df.head(top_n).copy()
    plot_df["group"] = plot_df["feature"].map(feature_group)
    fig = px.bar(
        plot_df.sort_values("importance"),
        x="importance",
        y="feature",
        color="group",
        orientation="h",
        title=f"Top {top_n} Features",
        labels={"importance": kind.replace("_", " ").title(), "feature": "Feature"},
        color_discrete_map=FEATURE_GROUP_COLORS,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=max(420, top_n * 24))
    return fig


def importance_table(slug: str, kind: str, top_n: int) -> pd.DataFrame:
    df = load_importance(slug, kind)
    if df.empty:
        return pd.DataFrame({"message": ["No explainability artifact found for this model."]})
    out = df.head(top_n).copy()
    out["feature_group"] = out["feature"].map(feature_group)
    out["importance"] = out["importance"].round(6)
    return out[["feature", "feature_group", "importance"]]


def model_options(metrics: dict, require_model_file: bool = False, require_explainability: bool = False):
    options = []
    for model in metrics.get("models", []):
        slug = model["slug"]
        if require_model_file and not (MODEL_DIR / f"{slug}.joblib").exists():
            continue
        if require_explainability and not (EXPLAIN_DIR / f"{slug}_feature_importance.csv").exists():
            continue
        options.append({"label": model["name"], "value": slug})
    return options


def default_model_slug(metrics: dict, require_model_file: bool = False, require_explainability: bool = False) -> str | None:
    opts = model_options(metrics, require_model_file=require_model_file, require_explainability=require_explainability)
    return opts[0]["value"] if opts else None


def categorical_options(prefix: str, feature_cols: list[str], fallback: list[str]):
    values = [c.replace(prefix, "", 1) for c in feature_cols if c.startswith(prefix)]
    values = sorted(set(values))
    if not values:
        values = fallback
    return [{"label": v.replace("_", " ").title(), "value": v} for v in values]


def categorical_values(prefix: str, feature_cols: list[str], fallback: list[str]):
    values = [c.replace(prefix, "", 1) for c in feature_cols if c.startswith(prefix)]
    values = sorted(v for v in set(values) if v not in {"UNKNOWN", "NA"})
    return values or fallback


def set_dummy(vector: dict[str, float], feature_cols: list[str], prefix: str, value: str | None) -> None:
    for col in feature_cols:
        if col.startswith(prefix):
            vector[col] = 0.0
    if value:
        col = prefix + value
        if col in vector:
            vector[col] = 1.0


def load_model(slug: str):
    path = MODEL_DIR / f"{slug}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def build_prediction_vector(model_df: pl.DataFrame | None, feature_cols: list[str], values: dict) -> np.ndarray:
    if model_df is None:
        base = {c: 0.0 for c in feature_cols}
    else:
        medians = model_df.select([pl.col(c).median().alias(c) for c in feature_cols]).row(0, named=True)
        base = {c: float(v or 0) for c, v in medians.items()}

    for key in [
        "num_primary_outcomes",
        "num_secondary_outcomes",
        "num_conditions",
        "num_drugs",
        "num_sites",
        "num_collaborators",
        "min_age_years",
        "max_age_years",
    ]:
        if key in base:
            base[key] = float(values.get(key) or 0)

    if "age_range_years" in base:
        base["age_range_years"] = max(base.get("max_age_years", 0) - base.get("min_age_years", 0), 0)
    if "has_fda_record" in base:
        base["has_fda_record"] = float(values.get("has_fda_record") or 0)
    if "is_fda_regulated_drug" in base:
        base["is_fda_regulated_drug"] = float(values.get("has_fda_record") or 0)
    if "has_dmc" in base:
        base["has_dmc"] = float(values.get("has_dmc") or 0)
    if "healthy_volunteers" in base:
        base["healthy_volunteers"] = float(values.get("healthy_volunteers") or 0)

    set_dummy(base, feature_cols, "phases_", values.get("phase"))
    set_dummy(base, feature_cols, "sponsor_class_", values.get("sponsor_class"))
    set_dummy(base, feature_cols, "application_type_", values.get("application_type"))
    set_dummy(base, feature_cols, "sex_", values.get("sex"))
    set_dummy(base, feature_cols, "masking_", values.get("masking"))

    return np.array([[base.get(c, 0.0) for c in feature_cols]], dtype=np.float32)


def prediction_figure(probabilities: np.ndarray):
    df = pd.DataFrame({"outcome": OUTCOME_ORDER, "probability": probabilities})
    fig = px.bar(
        df,
        x="outcome",
        y="probability",
        color="outcome",
        color_discrete_map=OUTCOME_COLORS,
        title="Predicted Outcome Probability",
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20), height=360)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def counterfactual_candidates(values: dict, feature_cols: list[str]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []

    for value in categorical_values("phases_", feature_cols, ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]):
        if value != values.get("phase"):
            candidates.append((f"Phase -> {value}", {"phase": value}))

    for value in categorical_values("sponsor_class_", feature_cols, ["INDUSTRY", "NIH", "OTHER"]):
        if value != values.get("sponsor_class"):
            candidates.append((f"Sponsor class -> {value}", {"sponsor_class": value}))

    for value in categorical_values("masking_", feature_cols, ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]):
        if value != values.get("masking"):
            candidates.append((f"Masking -> {value}", {"masking": value}))

    for value in categorical_values("application_type_", feature_cols, ["NDA", "ANDA", "BLA"]):
        if value != values.get("application_type"):
            candidates.append((f"Application type -> {value}", {"application_type": value}))

    numeric_steps = {
        "num_sites": [1, 5, 10, 25, 50],
        "num_primary_outcomes": [1, 2, 3, 5],
        "num_secondary_outcomes": [0, 2, 5, 10],
        "num_conditions": [1, 2, 4],
        "num_drugs": [1, 2, 3],
        "num_collaborators": [0, 1, 3, 5],
    }
    for key, options in numeric_steps.items():
        current = values.get(key)
        for option in options:
            if current != option:
                label = key.replace("num_", "").replace("_", " ").title()
                candidates.append((f"{label} -> {option}", {key: option}))

    for key, label in [
        ("has_dmc", "DMC"),
        ("healthy_volunteers", "Healthy volunteers"),
        ("has_fda_record", "FDA record"),
    ]:
        current = int(values.get(key) or 0)
        candidates.append((f"{label} -> {'Yes' if current == 0 else 'No'}", {key: 1 - current}))

    return candidates


def counterfactual_table(
    model,
    model_df: pl.DataFrame | None,
    feature_cols: list[str],
    values: dict,
    base_completed_prob: float,
    top_n: int = 8,
) -> pd.DataFrame:
    rows = []
    for label, updates in counterfactual_candidates(values, feature_cols):
        scenario = dict(values)
        scenario.update(updates)
        vector = build_prediction_vector(model_df, feature_cols, scenario)
        completed_prob = float(model.predict_proba(vector)[0][0])
        effect = completed_prob - base_completed_prob
        if effect > 0:
            rows.append(
                {
                    "Change": label,
                    "Completed Probability": f"{completed_prob:.1%}",
                    "Effect": f"+{effect:.1%}",
                }
            )

    if not rows:
        return pd.DataFrame({"Message": ["No single tested change increased Completed probability."]})

    return pd.DataFrame(rows).sort_values(
        "Effect",
        key=lambda s: s.str.replace("+", "", regex=False).str.replace("%", "", regex=False).astype(float),
        ascending=False,
    ).head(top_n)


def build_app() -> Dash:
    joined_df = load_joined_dataset()
    model_df = load_model_dataset()
    metrics = load_metrics()
    feature_cols = load_feature_columns()

    min_year = int(joined_df["start_year"].min())
    max_year = int(joined_df["start_year"].max())
    sponsor_options = [{"label": s, "value": s} for s in sorted(joined_df["sponsor_class"].drop_nulls().unique().to_list())]

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Clinical Trial Outcome Dashboard"

    models_available = len(metrics.get("models", []))
    total_trials = joined_df.height
    matched_pct = float(joined_df["has_fda_record"].cast(pl.Float64).mean() * 100)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Complete or Collapse?"),
                            html.Div("Clinical trial outcome modeling with trial design, sponsor, and OpenFDA signals.", className="subtitle"),
                            html.Div(
                                [
                                    html.Div("Developed by Zheng Xing and Andrew Wong", className="byline"),
                                    html.A(
                                        html.Img(
                                            src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                                            alt="GitHub repository",
                                            className="github-logo",
                                            style={
                                                "width": "18px",
                                                "height": "18px",
                                                "maxWidth": "18px",
                                                "maxHeight": "18px",
                                                "objectFit": "contain",
                                                "display": "block",
                                            },
                                        ),
                                        href="https://github.com/WuIlxy/245-Project",
                                        target="_blank",
                                        title="Open GitHub repository",
                                        className="github-button",
                                        style={
                                            "width": "26px",
                                            "height": "26px",
                                            "minWidth": "26px",
                                            "maxWidth": "26px",
                                            "minHeight": "26px",
                                            "overflow": "hidden",
                                        },
                                    ),
                                ],
                                className="byline-row",
                            ),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            metric_card("Joined Trials", f"{total_trials:,}", "drug-intervention subset"),
                            metric_card("OpenFDA Match", f"{matched_pct:.1f}%", "fuzzy drug-name join"),
                            metric_card("Models", str(models_available), "Logistic Regression / Random Forest / XGBoost"),
                        ],
                        className="metric-row",
                    ),
                ],
                className="header",
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview"),
                    dcc.Tab(label="Models", value="models"),
                    dcc.Tab(label="Explainability", value="explain"),
                    dcc.Tab(label="Prediction Demo", value="predict"),
                ],
            ),
            html.Div(id="tab-content", className="content"),
        ],
        className="page",
    )

    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        if tab == "overview":
            return html.Div(
                [
                    card(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Outcome"),
                                            dcc.Checklist(
                                                id="overview-outcomes",
                                                options=[{"label": o.title(), "value": o} for o in OUTCOME_ORDER],
                                                value=OUTCOME_ORDER,
                                                inline=True,
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Sponsor"),
                                            dcc.Dropdown(
                                                id="overview-sponsors",
                                                options=sponsor_options,
                                                value=[],
                                                multi=True,
                                                placeholder="All sponsors",
                                            ),
                                        ],
                                        className="control wide",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("FDA Match"),
                                            dcc.RadioItems(
                                                id="overview-fda",
                                                options=[
                                                    {"label": "All", "value": "all"},
                                                    {"label": "Matched", "value": "matched"},
                                                    {"label": "Unmatched", "value": "unmatched"},
                                                ],
                                                value="all",
                                                inline=True,
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Start Year"),
                                            dcc.RangeSlider(
                                                id="overview-years",
                                                min=min_year,
                                                max=max_year,
                                                value=[max(2000, min_year), min(2025, max_year)],
                                                marks={year: str(year) for year in range(max(2000, min_year), max_year + 1, 5)},
                                                step=1,
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ],
                                        className="control full",
                                    ),
                                ],
                                className="controls-grid",
                            )
                        ]
                    ),
                    html.Div(
                        [
                            card(dcc.Graph(id="outcome-chart")),
                            card(dcc.Graph(id="fda-chart")),
                            card(dcc.Graph(id="sponsor-chart"), "card span-2"),
                            card(dcc.Graph(id="enrollment-chart"), "card span-2"),
                        ],
                        className="grid",
                    ),
                ]
            )

        if tab == "models":
            opts = model_options(metrics)
            return html.Div(
                [
                    html.Div(
                        [
                            card(dcc.Graph(figure=model_metric_figure(metrics)), "card span-2"),
                            card(
                                [
                                    html.Label("Model"),
                                    dcc.Dropdown(
                                        id="model-report-dropdown",
                                        options=opts,
                                        value=default_model_slug(metrics),
                                        clearable=False,
                                    ),
                                    dcc.Graph(id="confusion-matrix"),
                                ],
                                "card",
                            ),
                            card(html.Div(id="classification-report-table"), "card span-2"),
                        ],
                        className="grid",
                    )
                ]
            )

        if tab == "explain":
            opts = model_options(metrics, require_explainability=True)
            return html.Div(
                [
                    card(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Model"),
                                            dcc.Dropdown(
                                                id="importance-model",
                                                options=opts,
                                                value=default_model_slug(metrics, require_explainability=True),
                                                clearable=False,
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Explanation Type"),
                                            dcc.RadioItems(
                                                id="importance-kind",
                                                options=[
                                                    {"label": "Tree importance", "value": "feature_importance"},
                                                    {"label": "Permutation F1 impact", "value": "permutation_importance"},
                                                    {"label": "SHAP summary", "value": "shap_summary"},
                                                ],
                                                value="feature_importance",
                                                inline=True,
                                            ),
                                        ],
                                        className="control wide",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Top Features"),
                                            dcc.Slider(id="importance-top-n", min=5, max=30, step=5, value=15),
                                        ],
                                        className="control full",
                                    ),
                                ],
                                className="controls-grid",
                            )
                        ]
                    ),
                    html.Div(
                        [
                            card(dcc.Graph(id="importance-chart"), "card span-2"),
                            card(html.Div(id="importance-table"), "card span-2"),
                        ],
                        className="grid",
                    ),
                ]
            )

        return html.Div(
            [
                card(
                    [
                        html.Div(
                            [
                                html.Div([html.Label("Model"), dcc.Dropdown(id="prediction-model", options=model_options(metrics, require_model_file=True), value=default_model_slug(metrics, require_model_file=True), clearable=False)], className="control"),
                                html.Div([html.Label("Phase"), dcc.Dropdown(id="pred-phase", options=categorical_options("phases_", feature_cols, ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]), value="PHASE2")], className="control"),
                                html.Div([html.Label("Sponsor Class"), dcc.Dropdown(id="pred-sponsor", options=categorical_options("sponsor_class_", feature_cols, ["INDUSTRY", "NIH", "OTHER"]), value="INDUSTRY")], className="control"),
                                html.Div([html.Label("Sex"), dcc.Dropdown(id="pred-sex", options=categorical_options("sex_", feature_cols, ["ALL", "FEMALE", "MALE"]), value="ALL")], className="control"),
                                html.Div([html.Label("Masking"), dcc.Dropdown(id="pred-masking", options=categorical_options("masking_", feature_cols, ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]), value="DOUBLE")], className="control"),
                                html.Div([html.Label("Application Type"), dcc.Dropdown(id="pred-application", options=categorical_options("application_type_", feature_cols, ["NDA", "ANDA", "BLA"]), value="NDA")], className="control"),
                                html.Div([html.Label("Sites"), dcc.Input(id="pred-sites", type="number", min=0, value=5, step=1)], className="control"),
                                html.Div([html.Label("Primary Outcomes"), dcc.Input(id="pred-primary", type="number", min=0, value=1, step=1)], className="control"),
                                html.Div([html.Label("Secondary Outcomes"), dcc.Input(id="pred-secondary", type="number", min=0, value=2, step=1)], className="control"),
                                html.Div([html.Label("Conditions"), dcc.Input(id="pred-conditions", type="number", min=0, value=1, step=1)], className="control"),
                                html.Div([html.Label("Drug Count"), dcc.Input(id="pred-drugs", type="number", min=0, value=1, step=1)], className="control"),
                                html.Div([html.Label("Collaborators"), dcc.Input(id="pred-collaborators", type="number", min=0, value=0, step=1)], className="control"),
                                html.Div([html.Label("Minimum Age"), dcc.Input(id="pred-min-age", type="number", min=0, value=18, step=1)], className="control"),
                                html.Div([html.Label("Maximum Age"), dcc.Input(id="pred-max-age", type="number", min=0, value=75, step=1)], className="control"),
                                html.Div(
                                    [
                                        html.Label("Flags"),
                                        dcc.Checklist(
                                            id="pred-flags",
                                            options=[
                                                {"label": "FDA record", "value": "has_fda_record"},
                                                {"label": "DMC", "value": "has_dmc"},
                                                {"label": "Healthy volunteers", "value": "healthy_volunteers"},
                                            ],
                                            value=["has_fda_record"],
                                            inline=True,
                                        ),
                                    ],
                                    className="control full",
                                ),
                                html.Button("Predict", id="predict-button", n_clicks=0, className="primary-button"),
                            ],
                            className="controls-grid",
                        )
                    ]
                ),
                dcc.Loading(
                    id="prediction-loading",
                    type="circle",
                    color="#047D95",
                    children=html.Div(id="prediction-output"),
                ),
            ]
        )

    @app.callback(
        Output("outcome-chart", "figure"),
        Output("fda-chart", "figure"),
        Output("sponsor-chart", "figure"),
        Output("enrollment-chart", "figure"),
        Input("overview-outcomes", "value"),
        Input("overview-sponsors", "value"),
        Input("overview-fda", "value"),
        Input("overview-years", "value"),
    )
    def update_overview(outcomes, sponsors, fda_filter, year_range):
        df = filter_joined(joined_df, outcomes, sponsors, fda_filter, year_range)
        if df.height == 0:
            blank = go.Figure()
            blank.add_annotation(text="No trials match the selected filters.", x=0.5, y=0.5, showarrow=False)
            return blank, blank, blank, blank
        return (
            outcome_distribution_figure(df),
            fda_match_figure(df),
            sponsor_outcome_figure(df),
            enrollment_figure(df),
        )

    @app.callback(
        Output("confusion-matrix", "figure"),
        Output("classification-report-table", "children"),
        Input("model-report-dropdown", "value"),
    )
    def update_model_report(slug):
        return confusion_matrix_figure(metrics, slug), make_table(report_table(metrics, slug), page_size=8)

    @app.callback(
        Output("importance-chart", "figure"),
        Output("importance-table", "children"),
        Input("importance-model", "value"),
        Input("importance-kind", "value"),
        Input("importance-top-n", "value"),
    )
    def update_importance(slug, kind, top_n):
        return importance_figure(slug, kind, top_n), make_table(importance_table(slug, kind, top_n), page_size=top_n)

    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-button", "n_clicks"),
        State("prediction-model", "value"),
        State("pred-phase", "value"),
        State("pred-sponsor", "value"),
        State("pred-sex", "value"),
        State("pred-masking", "value"),
        State("pred-application", "value"),
        State("pred-sites", "value"),
        State("pred-primary", "value"),
        State("pred-secondary", "value"),
        State("pred-conditions", "value"),
        State("pred-drugs", "value"),
        State("pred-collaborators", "value"),
        State("pred-min-age", "value"),
        State("pred-max-age", "value"),
        State("pred-flags", "value"),
    )
    def update_prediction(
        n_clicks,
        slug,
        phase,
        sponsor,
        sex,
        masking,
        application_type,
        sites,
        primary,
        secondary,
        conditions,
        drugs,
        collaborators,
        min_age,
        max_age,
        flags,
    ):
        if not n_clicks:
            raise PreventUpdate

        if not slug or not feature_cols:
            fig = go.Figure()
            fig.add_annotation(text="Run the model notebooks to enable predictions.", x=0.5, y=0.5, showarrow=False)
            return html.Div([card(dcc.Graph(figure=fig), "card span-2")], className="grid")

        model = load_model(slug)
        if model is None:
            fig = go.Figure()
            fig.add_annotation(text="Selected model file is missing.", x=0.5, y=0.5, showarrow=False)
            return html.Div([card(dcc.Graph(figure=fig), "card span-2")], className="grid")

        flags = flags or []
        scenario_values = {
            "phase": phase,
            "sponsor_class": sponsor,
            "sex": sex,
            "masking": masking,
            "application_type": application_type,
            "num_sites": sites,
            "num_primary_outcomes": primary,
            "num_secondary_outcomes": secondary,
            "num_conditions": conditions,
            "num_drugs": drugs,
            "num_collaborators": collaborators,
            "min_age_years": min_age,
            "max_age_years": max_age,
            "has_fda_record": int("has_fda_record" in flags),
            "has_dmc": int("has_dmc" in flags),
            "healthy_volunteers": int("healthy_volunteers" in flags),
        }
        vector = build_prediction_vector(
            model_df,
            feature_cols,
            scenario_values,
        )
        probabilities = model.predict_proba(vector)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = LABEL_NAMES[predicted_idx]
        counterfactuals = counterfactual_table(
            model=model,
            model_df=model_df,
            feature_cols=feature_cols,
            values=scenario_values,
            base_completed_prob=float(probabilities[0]),
        )
        summary = html.Div(
            [
                html.H3(f"Predicted: {predicted_label.title()}"),
                html.P(f"Confidence: {probabilities[predicted_idx]:.1%}"),
                html.P("Scenario prediction from the model-ready feature set."),
            ],
            className="prediction-summary",
        )
        return html.Div(
            [
                card(dcc.Graph(figure=prediction_figure(probabilities)), "card"),
                card(summary, "card"),
                card(
                    [
                        html.H3("Counterfactual Sensitivity"),
                        html.P(
                            "These are the minimal changes to the trial that can increase Completed probability.",
                            className="counterfactual-note",
                        ),
                        make_table(counterfactuals, page_size=8),
                    ],
                    "card span-2",
                ),
            ],
            className="grid",
        )

    return app


app = build_app()


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Nunito, Avenir Next, Aptos, Segoe UI, Arial, sans-serif;
                background: #EAF7F9;
                color: #102A43;
            }
            .page {
                min-height: 100vh;
                --bg: #EAF7F9;
                --surface: #FFFFFF;
                --surface-soft: #F3FBFC;
                --border: #B9DDE5;
                --text: #102A43;
                --muted: #577285;
                --accent: #047D95;
                --accent-strong: #025E73;
                --accent-soft: #D8F3F6;
                --shadow: rgba(18, 83, 105, 0.10);
                background:
                    linear-gradient(180deg, rgba(216, 243, 246, 0.95), rgba(234, 247, 249, 0.95) 260px),
                    var(--bg);
                color: var(--text);
                transition: background 160ms ease, color 160ms ease;
            }
            .header {
                padding: 28px 36px 22px;
                background: var(--surface);
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                gap: 24px;
                align-items: center;
                box-shadow: 0 8px 30px var(--shadow);
            }
            h1 { margin: 0; font-size: 31px; letter-spacing: 0; font-weight: 850; color: var(--text); }
            .subtitle { color: var(--muted); margin-top: 8px; font-size: 15px; }
            .byline-row {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                margin-top: 10px;
            }
            .byline {
                display: inline-block;
                padding: 5px 7px 5px 10px;
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent-strong);
                font-size: 13px;
                font-weight: 750;
            }
            .metric-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: stretch; justify-content: flex-end; }
            .metric-card {
                min-width: 150px;
                padding: 12px 14px;
                border: 1px solid var(--border);
                background: var(--surface-soft);
                border-radius: 8px;
            }
            .metric-label { font-size: 12px; color: var(--muted); text-transform: uppercase; font-weight: 800; }
            .metric-value { font-size: 24px; font-weight: 800; margin-top: 4px; }
            .metric-note { font-size: 12px; color: var(--muted); margin-top: 2px; }
            .github-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 26px;
                height: 26px;
                min-width: 26px;
                max-width: 26px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: #FFFFFF;
                text-decoration: none;
                box-shadow: none;
                overflow: hidden;
            }
            .github-button:hover {
                border-color: var(--accent);
                background: var(--accent-soft);
                text-decoration: none;
            }
            .github-logo {
                width: 18px !important;
                height: 18px !important;
                max-width: 18px !important;
                max-height: 18px !important;
                object-fit: contain;
                display: block;
            }
            .content { padding: 22px 28px 32px; }
            .card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 8px 24px var(--shadow);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
                margin-top: 16px;
            }
            .span-2 { grid-column: span 2; }
            .controls-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(180px, 1fr));
                gap: 14px 18px;
                align-items: end;
            }
            .control label {
                display: block;
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 6px;
                color: var(--text);
            }
            .control.wide { grid-column: span 2; }
            .control.full { grid-column: 1 / -1; }
            input[type="number"] {
                width: 100%;
                min-height: 38px;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 14px;
                background: var(--surface);
                color: var(--text);
            }
            .primary-button {
                min-height: 40px;
                border: 0;
                border-radius: 6px;
                background: var(--accent);
                color: white;
                font-weight: 700;
                cursor: pointer;
                padding: 0 18px;
            }
            .primary-button:hover { background: var(--accent-strong); }
            #prediction-loading {
                margin-top: 16px;
                min-height: 120px;
            }
            .prediction-summary h3 { margin: 0 0 8px; font-size: 24px; }
            .prediction-summary p { margin: 8px 0; color: var(--muted); }
            .counterfactual-note { margin: 4px 0 12px; color: var(--muted); }
            .Select-control, .Select-menu-outer, .Select-value {
                background: var(--surface) !important;
                color: var(--text) !important;
                border-color: var(--border) !important;
            }
            .Select-placeholder, .Select--single > .Select-control .Select-value {
                color: var(--muted) !important;
            }
            input[type="checkbox"],
            input[type="radio"] {
                accent-color: var(--accent);
            }
            .tab {
                background: #D8F3F6 !important;
                color: #083B4C !important;
                border-color: #B9DDE5 !important;
                font-weight: 750;
            }
            .tab--selected {
                background: #FFFFFF !important;
                color: #025E73 !important;
                border-top: 3px solid var(--accent) !important;
            }
            .rc-slider-track {
                background-color: var(--accent) !important;
            }
            .rc-slider-handle {
                border-color: var(--accent) !important;
                background-color: var(--accent) !important;
                box-shadow: 0 0 0 4px rgba(4, 125, 149, 0.16) !important;
            }
            .rc-slider-handle:focus,
            .rc-slider-handle:hover,
            .rc-slider-handle:active {
                border-color: var(--accent-strong) !important;
                box-shadow: 0 0 0 5px rgba(4, 125, 149, 0.20) !important;
            }
            .rc-slider-dot-active {
                border-color: var(--accent) !important;
            }
            .rc-slider-mark-text-active {
                color: var(--accent-strong) !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
                color: var(--text) !important;
            }
            .js-plotly-plot .plotly .cursor-crosshair,
            .js-plotly-plot .plotly .cursor-move,
            .js-plotly-plot .plotly .cursor-pointer {
                cursor: pointer !important;
            }
            @media (max-width: 900px) {
                .header { flex-direction: column; align-items: flex-start; }
                .grid { grid-template-columns: 1fr; }
                .span-2 { grid-column: span 1; }
                .controls-grid { grid-template-columns: 1fr; }
                .control.wide, .control.full { grid-column: span 1; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=8050, dev_tools_ui=False, dev_tools_props_check=False)
