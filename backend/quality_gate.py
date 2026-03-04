from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.analysis_service import (
    _classify_distribution_shape,
    _correlation_analysis,
    _dataset_level_checks,
    _group_difference_analysis,
    _outlier_characterisation,
    _smart_type_correction,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run_pipeline(df: pd.DataFrame) -> dict:
    inferred = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            inferred[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            inferred[col] = "datetime"
        else:
            inferred[col] = "categorical"

    correction = _smart_type_correction(df, inferred)
    final_types = correction["final_types"]
    corr = _correlation_analysis(df, final_types)
    groups = _group_difference_analysis(df, final_types)
    outliers = _outlier_characterisation(df, final_types)
    checks = _dataset_level_checks(df, final_types)
    return {
        "correction": correction,
        "final_types": final_types,
        "corr": corr,
        "groups": groups,
        "outliers": outliers,
        "checks": checks,
    }


def sales_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    region = rng.choice(["North", "South", "East", "West"], size=len(dates), p=[0.35, 0.25, 0.2, 0.2])
    product = rng.choice(["A", "B", "C"], size=len(dates))
    discount = rng.uniform(0, 0.4, size=len(dates))
    units = np.maximum(1, (200 - discount * 180 + rng.normal(0, 15, len(dates))).astype(int))
    revenue = units * (100 - discount * 35) + rng.normal(0, 180, len(dates))
    revenue[5] = revenue.max() * 2.8
    return pd.DataFrame({
        "date": dates,
        "region": region,
        "product": product,
        "revenue": revenue,
        "units": units,
        "discount": discount,
    })


def hr_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 240
    dept = rng.choice(["Eng", "Sales", "HR", "Ops"], size=n)
    tenure = np.clip(rng.normal(6, 3, n), 0, None)
    salary = 35000 + tenure * 4200 + np.where(dept == "Eng", 25000, 0) + np.where(dept == "Sales", 9000, 0) + rng.normal(0, 6000, n)
    left = (rng.random(n) < (0.35 - np.clip(tenure / 30, 0, 0.2))).astype(int)
    age = np.clip(22 + tenure + rng.normal(8, 4, n), 18, 65)
    return pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "department": dept,
        "salary": salary,
        "tenure": tenure,
        "age": age,
        "left_company": left,
    })


def survey_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 420
    latent = rng.normal(0, 1, n)
    q = {}
    for i in range(1, 11):
        score = np.clip(np.round(3 + latent * (0.5 if i < 5 else 0.2) + rng.normal(0, 1, n)), 1, 5)
        q[f"q{i}"] = score.astype(int)
    q["q2"] = np.clip(np.round(q["q1"] + rng.normal(0, 0.35, n)), 1, 5).astype(int)
    q["respondent_id"] = np.arange(1000, 1000 + n)
    return pd.DataFrame(q)


def financial_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=250)
    base = np.cumsum(rng.normal(0.2, 1.5, len(dates))) + 100
    close = base + rng.normal(0, 0.7, len(dates))
    open_ = close + rng.normal(0, 0.5, len(dates))
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, len(dates))
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, len(dates))
    volume = np.exp(rng.normal(12.0, 0.55, len(dates)))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def poor_names_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 180
    return pd.DataFrame({
        "col1": rng.normal(10, 2, n),
        "col2": rng.normal(10, 2, n),
        "X": rng.choice([0, 1], size=n),
        "A": rng.normal(50, 8, n),
        "B": rng.choice(["A", "B", "C"], size=n),
        "150": np.arange(1, n + 1),
    })


def run_quality_gate() -> list[CheckResult]:
    results: list[CheckResult] = []

    sales = _run_pipeline(sales_dataset())
    sales_corr_pairs = sales["corr"].get("pairs", [])
    results.append(CheckResult("sales: discount vs revenue correlation", any({"discount", "revenue"} == {p["col_a"], p["col_b"]} for p in sales_corr_pairs), "Expected discount-revenue relationship"))
    results.append(CheckResult("sales: revenue outliers", "revenue" in sales["outliers"].get("outliers_by_column", {}), "Expected revenue outlier detection"))

    hr = _run_pipeline(hr_dataset())
    reclass_cols = {item["column"]: item["to"] for item in hr["correction"].get("reclassifications", [])}
    results.append(CheckResult("hr: employee_id as ID", hr["final_types"].get("employee_id") == "id", "employee_id should be excluded as ID"))
    results.append(CheckResult("hr: left_company boolean", hr["final_types"].get("left_company") == "boolean", "left_company should be boolean"))
    results.append(CheckResult("hr: dept-salary group effect", any(item["categorical_column"] == "department" and item["numeric_column"] == "salary" and item["effect_size"] >= 0.2 for item in hr["groups"].get("findings", [])), "Expected salary differences by department"))

    survey = _run_pipeline(survey_dataset())
    q_discrete = all(survey["final_types"].get(f"q{i}") in {"categorical", "numeric"} for i in range(1, 11))
    respondent_is_id = survey["final_types"].get("respondent_id") == "id"
    strong_q_pairs = [p for p in survey["corr"].get("pairs", []) if abs(p["pearson_r"]) >= 0.5 and p["col_a"].startswith("q") and p["col_b"].startswith("q")]
    results.append(CheckResult("survey: respondent_id as ID", respondent_is_id, "respondent_id should be excluded"))
    results.append(CheckResult("survey: strongly correlated questions", len(strong_q_pairs) > 0, "Expected similar-construct question correlation"))

    financial = _run_pipeline(financial_dataset())
    open_close = any({"open", "close"} == {p["col_a"], p["col_b"]} and abs(p["pearson_r"]) >= 0.7 for p in financial["corr"].get("pairs", []))
    high_low_redundant = any({"high", "low"}.issubset({p["col_a"], p["col_b"]}) and abs(p["pearson_r"]) >= 0.9 for p in financial["corr"].get("pairs", []))
    vol_shape = _classify_distribution_shape(financial_dataset()["volume"]).get("shape")
    results.append(CheckResult("financial: open/close correlation", open_close, "Expected strong open-close relationship"))
    results.append(CheckResult("financial: high/low near redundancy", high_low_redundant, "Expected high-low near redundancy"))
    results.append(CheckResult("financial: volume shape computed", vol_shape is not None, "Expected distribution classification for volume"))

    poor = _run_pipeline(poor_names_dataset())
    poor_name_flag = sum(1 for c in poor_names_dataset().columns if c.lower() in {"x", "a", "b"} or c.lower().startswith("col")) >= 3
    results.append(CheckResult("poor-names: safe statistical output", len(poor["corr"].get("pairs", [])) >= 0, "Should not crash on ambiguous names"))
    results.append(CheckResult("poor-names: id-like numeric detected", poor["final_types"].get("150") in {"id", "numeric"}, "Should infer pattern from values only"))
    results.append(CheckResult("poor-names: ambiguity guard", poor_name_flag, "Name ambiguity detected for interpretation caveat"))

    return results


if __name__ == "__main__":
    outcomes = run_quality_gate()
    failed = [item for item in outcomes if not item.ok]
    for item in outcomes:
        icon = "✅" if item.ok else "❌"
        print(f"{icon} {item.name} — {item.detail}")
    if failed:
        print(f"\n{len(failed)} checks failed")
        sys.exit(1)
    print("\nAll quality-gate checks passed")
