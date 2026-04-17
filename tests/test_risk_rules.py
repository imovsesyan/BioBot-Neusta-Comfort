import pandas as pd

from biobot.risk.rules import (
    add_livability_score_status,
    add_risk_labels,
    create_rule_alerts,
)


def test_add_risk_labels_assigns_ordered_humidex_bands():
    df = pd.DataFrame({"humidex_c": [20.0, 35.0, 42.0, 46.0, 55.0]})

    result = add_risk_labels(df)

    assert result["risk_level"].tolist() == [
        "livable",
        "discomfort",
        "high_risk",
        "dangerous",
        "dangerous",
    ]
    assert result["risk_score"].tolist() == [0, 1, 2, 3, 3]
    assert result["is_critical_humidex"].tolist() == [False, False, False, False, True]


def test_create_rule_alerts_ignores_livable_rows_and_marks_critical():
    df = pd.DataFrame({"humidex_c": [20.0, 35.0, 42.0, 55.0]})
    risk_df = add_risk_labels(df)

    alerts = create_rule_alerts(risk_df)

    assert alerts["alert_severity"].tolist() == ["info", "warning", "critical"]
    assert "livable" not in alerts["risk_level"].tolist()


def test_add_livability_score_status_uses_current_neusta_direction():
    df = pd.DataFrame({"vivabilite_binary_mean": [0.0, 0.49, 0.5, 1.0]})

    result = add_livability_score_status(df)

    assert result["livability_status"].tolist() == [
        "livable",
        "livable",
        "not_livable",
        "not_livable",
    ]
    assert result["is_livable_by_score"].tolist() == [True, True, False, False]
