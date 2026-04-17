"""Rule-based heat-risk labels and alerts for F10."""

from __future__ import annotations

import pandas as pd


RISK_LEVEL_ORDER = ["livable", "discomfort", "high_risk", "dangerous"]
LIVABILITY_STATUS_ORDER = ["livable", "not_livable"]
DEFAULT_LIVABILITY_THRESHOLD = 0.5

RISK_LEVEL_DETAILS = {
    "livable": {
        "risk_score": 0,
        "humidex_range": "humidex < 30",
        "meaning": "Little or no discomfort",
    },
    "discomfort": {
        "risk_score": 1,
        "humidex_range": "30 <= humidex < 40",
        "meaning": "Some discomfort",
    },
    "high_risk": {
        "risk_score": 2,
        "humidex_range": "40 <= humidex <= 45",
        "meaning": "Great discomfort; reduce exertion",
    },
    "dangerous": {
        "risk_score": 3,
        "humidex_range": "humidex > 45",
        "meaning": "Dangerous heat stress conditions",
    },
}


ALERT_DETAILS = {
    "discomfort": {
        "alert_severity": "info",
        "alert_type": "thermal_discomfort",
        "alert_message": "Some discomfort expected. Monitor exposure and hydration.",
    },
    "high_risk": {
        "alert_severity": "warning",
        "alert_type": "heat_stress_warning",
        "alert_message": "High heat stress risk. Reduce exertion and limit exposure.",
    },
    "dangerous": {
        "alert_severity": "danger",
        "alert_type": "dangerous_heat_alert",
        "alert_message": "Dangerous heat stress. Trigger protective actions immediately.",
    },
    "critical": {
        "alert_severity": "critical",
        "alert_type": "critical_heat_alert",
        "alert_message": "Critical humidex above 54. Immediate safety response required.",
    },
}


def assign_humidex_risk_level(humidex: pd.Series) -> pd.DataFrame:
    """Assign risk levels from humidex values."""

    values = pd.to_numeric(humidex, errors="coerce")
    risk_level = pd.Series(pd.NA, index=values.index, dtype="object")

    risk_level.loc[values < 30] = "livable"
    risk_level.loc[(values >= 30) & (values < 40)] = "discomfort"
    risk_level.loc[(values >= 40) & (values <= 45)] = "high_risk"
    risk_level.loc[values > 45] = "dangerous"

    risk_score = risk_level.map(
        {level: details["risk_score"] for level, details in RISK_LEVEL_DETAILS.items()}
    )
    risk_meaning = risk_level.map(
        {level: details["meaning"] for level, details in RISK_LEVEL_DETAILS.items()}
    )

    return pd.DataFrame(
        {
            "risk_level": risk_level,
            "risk_score": risk_score.astype("Int64"),
            "risk_meaning": risk_meaning,
            "is_livable": risk_level == "livable",
            "is_dangerous": risk_level == "dangerous",
            "is_critical_humidex": values > 54,
        }
    )


def add_risk_labels(df: pd.DataFrame, humidex_column: str = "humidex_c") -> pd.DataFrame:
    """Return a copy of a table with F10 humidex-based risk labels."""

    out = df.copy()
    risk_labels = assign_humidex_risk_level(out[humidex_column])
    for column in risk_labels.columns:
        out[column] = risk_labels[column].values
    return out


def assign_livability_score_status(
    score: pd.Series,
    threshold: float = DEFAULT_LIVABILITY_THRESHOLD,
    higher_score_means_risk: bool = True,
) -> pd.DataFrame:
    """Assign livable/not-livable status from a 0..1 score.

    In the current Neusta `vivabilite_binary_mean` data, higher values appear
    during warmer and less comfortable periods. Therefore the default
    interpretation is: score >= 0.5 means not livable.
    """

    values = pd.to_numeric(score, errors="coerce")
    status = pd.Series(pd.NA, index=values.index, dtype="object")

    if higher_score_means_risk:
        status.loc[values < threshold] = "livable"
        status.loc[values >= threshold] = "not_livable"
    else:
        status.loc[values >= threshold] = "livable"
        status.loc[values < threshold] = "not_livable"

    return pd.DataFrame(
        {
            "livability_score": values,
            "livability_status": status,
            "livability_status_score": status.map({"livable": 0, "not_livable": 1}).astype("Int64"),
            "is_livable_by_score": status == "livable",
            "is_not_livable_by_score": status == "not_livable",
        }
    )


def add_livability_score_status(
    df: pd.DataFrame,
    score_column: str = "vivabilite_binary_mean",
    threshold: float = DEFAULT_LIVABILITY_THRESHOLD,
    higher_score_means_risk: bool = True,
) -> pd.DataFrame:
    """Return a copy of a table with F10 livability score status columns."""

    out = df.copy()
    score_status = assign_livability_score_status(
        out[score_column],
        threshold=threshold,
        higher_score_means_risk=higher_score_means_risk,
    )
    for column in score_status.columns:
        out[column] = score_status[column].values
    return out


def create_rule_alerts(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Create rule-based alerts from risk-labeled rows."""

    alerts = risk_df[risk_df["risk_score"].fillna(0) > 0].copy()
    if alerts.empty:
        return alerts

    alerts["alert_rule"] = "humidex_threshold"
    alerts["alert_level"] = alerts["risk_level"].where(
        ~alerts["is_critical_humidex"],
        "critical",
    )
    alerts["alert_severity"] = alerts["alert_level"].map(
        {level: details["alert_severity"] for level, details in ALERT_DETAILS.items()}
    )
    alerts["alert_type"] = alerts["alert_level"].map(
        {level: details["alert_type"] for level, details in ALERT_DETAILS.items()}
    )
    alerts["alert_message"] = alerts["alert_level"].map(
        {level: details["alert_message"] for level, details in ALERT_DETAILS.items()}
    )
    return alerts


def risk_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count rows in each ordered F10 risk level."""

    counts = df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0)
    return {level: int(counts[level]) for level in RISK_LEVEL_ORDER}


def livability_status_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count rows in each ordered F10 livability score status."""

    counts = df["livability_status"].value_counts().reindex(LIVABILITY_STATUS_ORDER, fill_value=0)
    return {level: int(counts[level]) for level in LIVABILITY_STATUS_ORDER}
