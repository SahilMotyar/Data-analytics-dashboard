from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for thread safety
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, linregress, zscore

from app.core.config import settings
from app.core.storage import read_json, write_json

logger = logging.getLogger(__name__)

DATETIME_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$") ,
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
]
BOOL_VALUES = {"true", "false", "yes", "no", "1", "0"}


def _severity_rank(level: str) -> int:
    return {"good": 0, "warning": 1, "critical": 2, "na": -1}.get(level, -1)


def _missing_health(missing_pct: float) -> dict[str, str]:
    if missing_pct == 0:
        return {"status": "good", "dot": "🟢", "label": "No missing values"}
    if missing_pct <= 10:
        return {"status": "warning", "dot": "🟡", "label": "Some missing values"}
    return {"status": "critical", "dot": "🔴", "label": "High missing values"}


def _outlier_health(outliers_iqr_count: int, row_count: int) -> dict[str, str]:
    threshold = 0.05 * row_count
    if outliers_iqr_count == 0:
        return {"status": "good", "dot": "🟢", "label": "No notable outliers"}
    if outliers_iqr_count <= threshold:
        return {"status": "warning", "dot": "🟡", "label": "Some outliers"}
    return {"status": "critical", "dot": "🔴", "label": "Many outliers"}


def _skew_health(skewness: float | None) -> dict[str, str]:
    if skewness is None:
        return {"status": "na", "dot": "⚪", "label": "Not applicable"}
    value = abs(skewness)
    if value < 0.5:
        return {"status": "good", "dot": "🟢", "label": "Mostly balanced"}
    if value < 1.0:
        return {"status": "warning", "dot": "🟡", "label": "Slightly skewed"}
    return {"status": "critical", "dot": "🔴", "label": "Highly skewed"}


def _cardinality_health(unique_count: int | None) -> dict[str, str]:
    if unique_count is None:
        return {"status": "na", "dot": "⚪", "label": "Not applicable"}
    if unique_count <= 20:
        return {"status": "good", "dot": "🟢", "label": "Cardinality looks good"}
    if unique_count <= 50:
        return {"status": "warning", "dot": "🟡", "label": "High cardinality"}
    return {"status": "critical", "dot": "🔴", "label": "Very high cardinality"}


def _compute_column_health(stats: dict[str, Any], row_count: int) -> dict[str, Any]:
    inferred_type = stats.get("inferred_type")
    missing = _missing_health(float(stats.get("missing_pct", 0)))
    outlier = _outlier_health(int(stats.get("outliers_iqr_count", 0)), row_count) if inferred_type == "numeric" else {"status": "na", "dot": "⚪", "label": "Not applicable"}
    skew = _skew_health(stats.get("skewness")) if inferred_type == "numeric" else {"status": "na", "dot": "⚪", "label": "Not applicable"}
    unique_count = stats.get("cardinality") if inferred_type in {"categorical", "boolean", "id", "free_text"} else None
    cardinality = _cardinality_health(unique_count)

    levels = [missing["status"]]
    for item in [outlier, skew, cardinality]:
        if item["status"] != "na":
            levels.append(item["status"])

    overall = max(levels, key=_severity_rank)
    overall_map = {
        "good": {"status": "good", "dot": "🟢", "label": "Healthy"},
        "warning": {"status": "warning", "dot": "🟡", "label": "Needs attention"},
        "critical": {"status": "critical", "dot": "🔴", "label": "Critical attention"},
    }

    return {
        "missing": missing,
        "outliers": outlier,
        "distribution": skew,
        "cardinality": cardinality,
        "overall": overall_map[overall],
    }


def _style_axis(ax, subtitle: str) -> None:
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.2f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", fontsize=12, color="#475569")


def upload_dir(upload_id: str) -> Path:
    return settings.uploads_dir / upload_id


def output_dir(upload_id: str) -> Path:
    target = settings.outputs_dir / upload_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def metadata_path(upload_id: str) -> Path:
    return upload_dir(upload_id) / "metadata.json"


def analysis_path(upload_id: str) -> Path:
    return output_dir(upload_id) / "analysis.json"


def _load_raw_dataframe(upload_id: str, sheet_name: str | None = None) -> pd.DataFrame:
    meta = read_json(metadata_path(upload_id), {})
    file_name = meta.get("file_name", "")
    file_path = upload_dir(upload_id) / "original" / file_name
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    if file_name.lower().endswith((".xlsx", ".xls")):
        sheet = sheet_name or meta.get("active_sheet")
        return pd.read_excel(file_path, sheet_name=sheet)
    raise ValueError("Unsupported file format")


def _regex_datetime_ratio(series: pd.Series) -> float:
    values = series.dropna().astype(str)
    if values.empty:
        return 0.0
    matches = values.apply(lambda item: any(pattern.match(item.strip()) for pattern in DATETIME_PATTERNS))
    return float(matches.mean())


def _infer_column_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"

    as_str = non_null.astype(str).str.strip()
    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)

    if _regex_datetime_ratio(series) >= 0.8:
        return "datetime"

    numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if numeric_ratio > 0.95:
        return "numeric"

    bool_ratio = as_str.str.lower().isin(BOOL_VALUES).mean()
    if bool_ratio > 0.95:
        return "boolean"

    if unique_ratio > 0.95 and (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return "id"

    mean_len = as_str.str.len().mean()
    if mean_len > 50:
        return "free_text"

    return "categorical"


def _is_integer_like(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False
    return bool(np.all(np.isclose(clean, np.round(clean))))


def _looks_like_sequential(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or not _is_integer_like(clean):
        return False
    ordered = np.sort(clean.to_numpy())
    diffs = np.diff(ordered)
    if len(diffs) == 0:
        return False
    return bool(np.mean(diffs == 1) >= 0.95)


def _name_suggests_id(column_name: str) -> bool:
    name = column_name.lower()
    tokens = ["id", "index", "no", "num", "#", "row", "key", "ref", "code"]
    return any(token in name for token in tokens)


def _is_percentage_column(column_name: str, series: pd.Series) -> bool:
    name = column_name.lower()
    hint = any(token in name for token in ["pct", "percent", "rate", "ratio", "share", "proportion"])
    if not hint:
        return False
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False
    in_zero_one = bool((clean.between(0, 1)).all())
    in_zero_hundred = bool((clean.between(0, 100)).all())
    return in_zero_one or in_zero_hundred


def _smart_type_correction(dataframe: pd.DataFrame, inferred_types: dict[str, str]) -> dict[str, Any]:
    row_count = len(dataframe)
    final_types = inferred_types.copy()
    reclassifications: list[dict[str, Any]] = []
    excluded_columns: list[dict[str, str]] = []
    high_missing_flags: list[dict[str, Any]] = []
    constant_flags: list[dict[str, Any]] = []
    text_flags: list[dict[str, Any]] = []
    percentage_flags: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        missing_pct = round(float(series.isna().mean() * 100), 2)
        original_type = final_types.get(column, "categorical")

        if missing_pct > 40:
            high_missing_flags.append({
                "column": column,
                "missing_pct": missing_pct,
                "message": f"{column} has {missing_pct}% missing values; findings may be unstable.",
            })

        if unique_count <= 1:
            final_types[column] = "constant"
            constant_flags.append({
                "column": column,
                "message": "This column has only one value — carries no information.",
            })
            excluded_columns.append({"column": column, "reason": "constant column"})
            if original_type != "constant":
                reclassifications.append({
                    "column": column,
                    "from": original_type,
                    "to": "constant",
                    "reason": "unique_count == 1",
                    "message": "This column has only one value — carries no information.",
                })
            continue

        if unique_count == row_count and (_looks_like_sequential(series) or (_name_suggests_id(column) and unique_count == row_count)):
            final_types[column] = "id"
            excluded_columns.append({"column": column, "reason": "row identifier"})
            if original_type != "id":
                reclassifications.append({
                    "column": column,
                    "from": original_type,
                    "to": "id",
                    "reason": "unique_count == row_count and identifier pattern",
                    "message": f"Excluded {column} — looks like a row identifier, not a measurement.",
                })
            continue

        numeric_clean = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric_clean.notna().mean())

        if unique_count == 2 and numeric_ratio > 0.95:
            values = set(numeric_clean.dropna().astype(int).tolist())
            if values.issubset({0, 1}) or values.issubset({-1, 1}):
                final_types[column] = "boolean"
                if original_type != "boolean":
                    reclassifications.append({
                        "column": column,
                        "from": original_type,
                        "to": "boolean",
                        "reason": "binary integer values in {0,1} or {-1,1}",
                        "message": f"{column} has binary integer values — treating as boolean.",
                    })
                continue

        if original_type == "numeric" and unique_count <= 10 and _is_integer_like(series):
            final_types[column] = "categorical"
            reclassifications.append({
                "column": column,
                "from": original_type,
                "to": "categorical",
                "reason": "numeric with <=10 whole-number distinct values",
                "message": f"{column} contains only {unique_count} distinct whole-number values — treating as categories, not measurements.",
            })
            continue

        if original_type != "datetime" and _regex_datetime_ratio(series) >= 0.8:
            final_types[column] = "datetime"
            if original_type != "datetime":
                reclassifications.append({
                    "column": column,
                    "from": original_type,
                    "to": "datetime",
                    "reason": ">80% values match date patterns",
                    "message": f"{column} detected as date-like values — treating as datetime.",
                })
            continue

        if _is_percentage_column(column, series):
            percentage_flags.append({
                "column": column,
                "message": f"{column} appears to be a percentage/rate column; visual scales should use percent labels.",
            })

        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            mean_len = float(non_null.astype(str).str.len().mean()) if not non_null.empty else 0.0
            if mean_len > 50:
                final_types[column] = "free_text"
                text_flags.append({
                    "column": column,
                    "message": "This column contains free text — consider NLP analysis separately.",
                })

    return {
        "final_types": final_types,
        "reclassifications": reclassifications,
        "excluded_columns": excluded_columns,
        "high_missing_flags": high_missing_flags,
        "constant_flags": constant_flags,
        "free_text_flags": text_flags,
        "percentage_flags": percentage_flags,
    }


def _kde_mode_count(values: pd.Series) -> tuple[int, list[float]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 20:
        return 1, []
    arr = clean.to_numpy(dtype=float)
    x_grid = np.linspace(np.min(arr), np.max(arr), 256)
    try:
        kde = gaussian_kde(arr)
        y = kde(x_grid)
    except Exception:
        return 1, []

    peaks, _ = find_peaks(y)
    if len(peaks) <= 1:
        return 1, x_grid[peaks].tolist() if len(peaks) else []

    kept = [int(peaks[0])]
    for current in peaks[1:]:
        prev = kept[-1]
        left, right = sorted([prev, int(current)])
        valley = float(np.min(y[left:right + 1])) if right > left else float(y[left])
        lower_peak = float(min(y[prev], y[current]))
        if lower_peak <= 0:
            kept.append(int(current))
            continue
        valley_ratio = valley / lower_peak
        if valley_ratio < 0.2:
            kept.append(int(current))

    return len(kept), x_grid[kept].tolist()


def _classify_distribution_shape(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"shape": "unknown", "modes": 0}

    unique_count = int(clean.nunique())
    mean_val = float(clean.mean()) if len(clean) else 0.0
    std_val = float(clean.std()) if len(clean) else 0.0
    skewness = float(clean.skew()) if len(clean) > 2 else 0.0
    kurtosis = float(clean.kurtosis()) if len(clean) > 3 else 0.0

    if unique_count < 15 and _is_integer_like(clean):
        return {
            "shape": "discrete_ordinal",
            "modes": unique_count,
            "message": "Discrete/ordinal values detected; bar chart is preferred over histogram.",
        }

    if mean_val != 0 and abs(std_val / mean_val) < 0.01:
        return {
            "shape": "near_constant",
            "modes": 1,
            "message": "Almost no variation — may not be useful.",
        }

    modes, mode_positions = _kde_mode_count(clean)

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        uniform_like = False
    else:
        hist, _ = np.histogram(clean, bins=min(12, max(5, unique_count // 3)))
        uniform_like = float(np.std(hist) / max(np.mean(hist), 1e-9)) < 0.2

    if uniform_like:
        return {"shape": "uniform", "modes": modes, "mode_positions": mode_positions, "message": "Values are roughly flat/uniform."}
    if modes >= 3:
        return {
            "shape": "multimodal",
            "modes": modes,
            "mode_positions": mode_positions,
            "message": "Multiple distinct subgroups detected.",
            "priority": "high",
        }
    if modes == 2:
        return {
            "shape": "bimodal",
            "modes": 2,
            "mode_positions": mode_positions,
            "message": "Two subgroups may be hidden in this column.",
            "priority": "high",
        }
    if kurtosis > 3:
        return {"shape": "unimodal_heavy_tailed", "modes": 1, "message": "Leptokurtic/heavy-tailed distribution."}
    if abs(skewness) >= 0.5:
        direction = "right" if skewness > 0 else "left"
        return {"shape": "unimodal_skewed", "modes": 1, "skew_direction": direction, "message": f"Unimodal but {direction}-skewed."}
    if -1 <= kurtosis <= 1 and abs(skewness) < 0.5:
        return {"shape": "unimodal_normal", "modes": 1, "message": "Unimodal and approximately symmetric."}
    return {"shape": "unimodal", "modes": 1}


def _correlation_analysis(dataframe: pd.DataFrame, final_types: dict[str, str], high_missing_threshold: float = 30.0) -> dict[str, Any]:
    eligible_numeric = []
    ordinal_candidates = []
    for col, typ in final_types.items():
        if typ == "numeric":
            pass
        elif typ == "categorical":
            numeric_col = pd.to_numeric(dataframe[col], errors="coerce")
            if numeric_col.notna().mean() > 0.95 and _is_integer_like(numeric_col) and int(numeric_col.dropna().nunique()) <= 15:
                ordinal_candidates.append(col)
                continue
            else:
                continue
        else:
            continue
        if round(float(dataframe[col].isna().mean() * 100), 2) > high_missing_threshold:
            continue
        eligible_numeric.append(col)

    eligible_all = list(dict.fromkeys(eligible_numeric + ordinal_candidates))

    pairs: list[dict[str, Any]] = []
    redundant_pairs: list[dict[str, Any]] = []
    notable_negative: list[dict[str, Any]] = []
    nonlinear_pairs: list[dict[str, Any]] = []

    for idx, col_a in enumerate(eligible_all):
        for col_b in eligible_all[idx + 1:]:
            pair_df = dataframe[[col_a, col_b]].copy()
            pair_df[col_a] = pd.to_numeric(pair_df[col_a], errors="coerce")
            pair_df[col_b] = pd.to_numeric(pair_df[col_b], errors="coerce")
            pair_df = pair_df.dropna()
            if len(pair_df) < 5:
                continue
            pearson = float(pair_df[col_a].corr(pair_df[col_b], method="pearson"))
            spearman = float(pair_df[col_a].corr(pair_df[col_b], method="spearman"))
            abs_r = abs(pearson)
            if abs_r >= 0.9:
                strength = "very_strong"
            elif abs_r >= 0.7:
                strength = "strong"
            elif abs_r >= 0.5:
                strength = "moderate"
            elif abs_r < 0.3:
                strength = "weak_none"
            else:
                strength = "weak"
            item = {
                "col_a": col_a,
                "col_b": col_b,
                "pearson_r": round(pearson, 4),
                "spearman_rho": round(spearman, 4),
                "strength": strength,
                "negative": pearson <= -0.5,
                "non_linear_signal": abs(spearman - pearson) > 0.2,
                "includes_ordinal": col_a in ordinal_candidates or col_b in ordinal_candidates,
            }
            pairs.append(item)

            if abs_r > 0.95:
                redundant_pairs.append(item)
            if pearson <= -0.5:
                notable_negative.append(item)
            if abs(spearman - pearson) > 0.2:
                nonlinear_pairs.append(item)

    pairs_sorted = sorted(pairs, key=lambda item: abs(item["pearson_r"]), reverse=True)
    return {
        "eligible_numeric_columns": eligible_numeric,
        "ordinal_columns": ordinal_candidates,
        "pairs": pairs_sorted,
        "top_positive": [item for item in pairs_sorted if item["pearson_r"] > 0][:3],
        "notable_negative": notable_negative,
        "redundant_pairs": redundant_pairs,
        "non_linear_pairs": nonlinear_pairs,
    }


def _group_difference_analysis(dataframe: pd.DataFrame, final_types: dict[str, str]) -> dict[str, Any]:
    categorical_cols = [col for col, typ in final_types.items() if typ in {"categorical", "boolean"}]
    numeric_cols = [col for col, typ in final_types.items() if typ == "numeric"]
    findings: list[dict[str, Any]] = []
    strongest_by_category: list[dict[str, Any]] = []

    for cat_col in categorical_cols:
        cat_best: dict[str, Any] | None = None
        for num_col in numeric_cols:
            sub = dataframe[[cat_col, num_col]].copy()
            sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
            sub = sub.dropna()
            if sub.empty:
                continue
            grouped = sub.groupby(cat_col)[num_col].agg(["mean", "std", "count"]).reset_index()
            if len(grouped) < 2:
                continue
            overall_std = float(sub[num_col].std())
            if overall_std == 0 or np.isnan(overall_std):
                effect_size = 0.0
            else:
                effect_size = float((grouped["mean"].max() - grouped["mean"].min()) / overall_std)

            if effect_size >= 1.0:
                effect_label = "large"
            elif effect_size >= 0.5:
                effect_label = "medium"
            elif effect_size >= 0.2:
                effect_label = "small"
            else:
                effect_label = "similar"

            variance_alerts = []
            std_nonzero = grouped[grouped["std"] > 0]
            if len(std_nonzero) >= 2:
                max_std_row = std_nonzero.loc[std_nonzero["std"].idxmax()]
                min_std_row = std_nonzero.loc[std_nonzero["std"].idxmin()]
                if float(max_std_row["std"]) >= 3 * float(min_std_row["std"]):
                    variance_alerts.append(
                        {
                            "message": f"Values are much more spread out in {max_std_row[cat_col]} than {min_std_row[cat_col]}.",
                            "max_group": str(max_std_row[cat_col]),
                            "min_group": str(min_std_row[cat_col]),
                            "ratio": round(float(max_std_row["std"] / min_std_row["std"]), 2),
                        }
                    )

            item = {
                "categorical_column": cat_col,
                "numeric_column": num_col,
                "effect_size": round(effect_size, 4),
                "effect_label": effect_label,
                "group_means": [
                    {
                        "group": str(row[cat_col]),
                        "mean": round(float(row["mean"]), 4),
                        "std": round(float(row["std"]), 4) if not np.isnan(float(row["std"])) else None,
                        "count": int(row["count"]),
                    }
                    for _, row in grouped.iterrows()
                ],
                "variance_alerts": variance_alerts,
            }
            findings.append(item)
            if cat_best is None or item["effect_size"] > cat_best["effect_size"]:
                cat_best = item

        if cat_best is not None:
            strongest_by_category.append(cat_best)

    return {
        "findings": sorted(findings, key=lambda item: item["effect_size"], reverse=True),
        "strongest_by_category": strongest_by_category,
    }


def _outlier_characterisation(dataframe: pd.DataFrame, final_types: dict[str, str]) -> dict[str, Any]:
    numeric_cols = [col for col, typ in final_types.items() if typ == "numeric"]
    outliers_by_column: dict[str, Any] = {}
    row_hits: dict[int, list[str]] = {}

    for col in numeric_cols:
        series = pd.to_numeric(dataframe[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        outlier_rows = dataframe.index[mask.fillna(False)].tolist()
        outlier_values = series[mask.fillna(False)]
        if len(outlier_rows) == 0:
            continue

        mean_v = float(clean.mean())
        std_v = float(clean.std()) if float(clean.std()) != 0 else 1e-9
        p99 = float(clean.quantile(0.99))

        details = []
        for idx, value in outlier_values.items():
            z = float((value - mean_v) / std_v)
            percentile = round(float((clean <= value).mean() * 100), 2)
            likely_error = bool(value > 3 * p99) if p99 > 0 else False
            details.append(
                {
                    "row_index": int(idx),
                    "value": float(value),
                    "z_score": round(z, 3),
                    "percentile": percentile,
                    "likely_data_error": likely_error,
                    "likely_genuine_extreme": not likely_error,
                }
            )
            row_hits.setdefault(int(idx), []).append(col)

        outlier_pct = round(float((len(outlier_rows) / max(len(dataframe), 1)) * 100), 2)
        summary_payload: dict[str, Any]
        if len(details) <= 10:
            summary_payload = {"listed_values": details}
        else:
            vals = [item["value"] for item in details]
            summary_payload = {
                "count": len(details),
                "min_outlier": float(min(vals)),
                "max_outlier": float(max(vals)),
                "outlier_pct": outlier_pct,
            }

        outliers_by_column[col] = {
            "count": len(details),
            "outlier_pct": outlier_pct,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "details": details,
            "summary": summary_payload,
        }

    multi_column_anomalies = [
        {"row_index": row_idx, "columns": cols}
        for row_idx, cols in row_hits.items()
        if len(cols) >= 2
    ]
    return {"outliers_by_column": outliers_by_column, "multi_column_anomalies": multi_column_anomalies}


def _dataset_level_checks(dataframe: pd.DataFrame, final_types: dict[str, str]) -> dict[str, Any]:
    duplicate_columns: list[dict[str, str]] = []
    cols = list(dataframe.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            if dataframe[col_a].equals(dataframe[col_b]):
                duplicate_columns.append({"col_a": col_a, "col_b": col_b, "message": "Columns are exactly identical."})

    sample_size_flags = []
    if len(dataframe) < 30:
        sample_size_flags.append("Warning: Very small dataset. Statistical patterns may not be reliable.")
    elif len(dataframe) < 100:
        sample_size_flags.append("Small dataset — correlations and distributions may shift with more data.")

    class_imbalance = []
    for col, typ in final_types.items():
        if typ not in {"categorical", "boolean"}:
            continue
        counts = dataframe[col].value_counts(dropna=True)
        if counts.empty:
            continue
        top_val = str(counts.index[0])
        top_pct = round(float((counts.iloc[0] / counts.sum()) * 100), 2)
        if top_pct > 80:
            class_imbalance.append(
                {
                    "column": col,
                    "top_category": top_val,
                    "top_pct": top_pct,
                    "message": f"Heavy imbalance: {top_val} makes up {top_pct}% of rows.",
                }
            )

    datetime_checks = []
    dt_cols = [col for col, typ in final_types.items() if typ == "datetime"]
    num_cols = [col for col, typ in final_types.items() if typ == "numeric"]
    target_col = num_cols[0] if num_cols else None
    for col in dt_cols:
        dt = pd.to_datetime(dataframe[col], errors="coerce").dropna().sort_values()
        if dt.empty:
            continue
        diffs = dt.diff().dropna()
        if diffs.empty:
            continue
        med_days = max(int(diffs.median().days), 1)
        if med_days <= 1:
            freq = "D"
            expected_label = "daily"
        elif med_days <= 7:
            freq = "W"
            expected_label = "weekly"
        else:
            freq = "M"
            expected_label = "monthly"
        trend = dt.to_frame(name=col).set_index(col).resample(freq).size()
        full_index = pd.date_range(start=trend.index.min(), end=trend.index.max(), freq=freq)
        missing_periods = full_index.difference(trend.index)

        trend_direction = "unknown"
        trend_r2 = None
        if target_col:
            temp = dataframe[[col, target_col]].copy()
            temp[col] = pd.to_datetime(temp[col], errors="coerce")
            temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
            temp = temp.dropna().sort_values(col)
            if len(temp) > 10:
                temp = temp.set_index(col).resample(freq)[target_col].mean().dropna()
                if len(temp) > 3:
                    x = np.arange(len(temp))
                    slope, intercept, r_value, p_value, std_err = linregress(x, temp.values)
                    r2 = float(r_value ** 2)
                    trend_r2 = round(r2, 4)
                    if r2 > 0.3:
                        trend_direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
                    else:
                        trend_direction = "flat"

        datetime_checks.append(
            {
                "column": col,
                "expected_frequency": expected_label,
                "missing_period_count": int(len(missing_periods)),
                "missing_periods": [str(item) for item in missing_periods[:100]],
                "range_start": str(dt.min()),
                "range_end": str(dt.max()),
                "target_trend_direction": trend_direction,
                "target_trend_r2": trend_r2,
            }
        )

    return {
        "duplicate_columns": duplicate_columns,
        "datetime_checks": datetime_checks,
        "sample_size_flags": sample_size_flags,
        "class_imbalance": class_imbalance,
    }


def _coerce_clean_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = dataframe.copy()
    mismatches: dict[str, str] = {}

    for column in cleaned.columns:
        series = cleaned[column]
        if pd.api.types.is_object_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_ratio = numeric.notna().mean()
            if numeric_ratio > 0.95:
                cleaned[column] = numeric
                continue

            dt = pd.to_datetime(series, errors="coerce", dayfirst=False)
            dt_ratio = dt.notna().mean()
            if dt_ratio > 0.8:
                cleaned[column] = dt
                continue

            mixed_numeric_and_text = numeric.notna().sum() > 0 and series.astype(str).str.strip().ne("").sum() > numeric.notna().sum()
            if mixed_numeric_and_text:
                mismatches[column] = "mixed_numeric_string"

    return cleaned, {"type_mismatches": mismatches}


def _quality_flags(dataframe: pd.DataFrame, inferred_types: dict[str, str]) -> dict[str, Any]:
    missing = {}
    constant_columns = []
    high_cardinality = []
    potential_ids = []

    for column in dataframe.columns:
        series = dataframe[column]
        miss_count = int(series.isna().sum())
        miss_pct = round(float((miss_count / max(len(dataframe), 1)) * 100), 2)
        missing[column] = {"count": miss_count, "pct": miss_pct, "warn": miss_pct > 5}

        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            constant_columns.append(column)

        if inferred_types.get(column) == "categorical" and unique_count > 50:
            high_cardinality.append(column)

        uniqueness_ratio = unique_count / max(len(series.dropna()), 1) if len(series.dropna()) else 0
        if uniqueness_ratio > 0.95:
            potential_ids.append(column)

    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_pct = round(float((duplicate_count / max(len(dataframe), 1)) * 100), 2)

    return {
        "missing": missing,
        "duplicates": {"count": duplicate_count, "pct": duplicate_pct},
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality,
        "potential_id_columns": potential_ids,
    }


def _dataset_summary(dataframe: pd.DataFrame, inferred_types: dict[str, str], flags: dict[str, Any]) -> dict[str, Any]:
    rows, cols = dataframe.shape
    memory_mb = round(float(dataframe.memory_usage(deep=True).sum() / (1024 * 1024)), 2)

    type_breakdown: dict[str, int] = {}
    for value in inferred_types.values():
        type_breakdown[value] = type_breakdown.get(value, 0) + 1

    missing_pct_avg = np.mean([item["pct"] for item in flags["missing"].values()]) if flags["missing"] else 0.0
    completeness_score = max(0.0, 100 - missing_pct_avg)
    consistency_penalty = len(flags["constant_columns"]) * 2 + flags["duplicates"]["pct"]
    consistency_score = max(0.0, 100 - consistency_penalty)
    uniqueness_penalty = len(flags["potential_id_columns"]) * 1.5
    uniqueness_score = max(0.0, 100 - uniqueness_penalty)

    quality_score = round(float((completeness_score + consistency_score + uniqueness_score) / 3), 2)

    return {
        "rows": int(rows),
        "columns": int(cols),
        "memory_mb": memory_mb,
        "type_breakdown": type_breakdown,
        "quality_score": quality_score,
    }


def _safe_chart_name(column_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", column_name)


def _save_hist_kde_box(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, str]:
    chart_base = _safe_chart_name(column)
    out_dir = output_dir(upload_id)
    hist_path = out_dir / f"{chart_base}_hist.png"
    box_path = out_dir / f"{chart_base}_box.png"

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mean_v = float(values.mean()) if not values.empty else 0.0
    median_v = float(values.median()) if not values.empty else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}", fontsize=14)
    _style_axis(ax, "A wider spread means more variation in values")
    ax.axvline(mean_v, color="orange", linestyle="--", linewidth=2, label=f"Mean: {mean_v:,.2f}")
    ax.axvline(median_v, color="green", linestyle="--", linewidth=2, label=f"Median: {median_v:,.2f}")
    ax.axvspan(values.min(), lower, color="#fecaca", alpha=0.25, label="Outlier zone")
    ax.axvspan(upper, values.max(), color="#fecaca", alpha=0.25)
    if median_v and abs(mean_v - median_v) / abs(median_v) > 0.1:
        ax.annotate(
            "Mean is pulled by outliers — median may better represent typical value",
            xy=(0.5, -0.22),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
        )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(hist_path)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x=values, ax=ax)
    ax.set_title(f"Box view of {column}", fontsize=14)
    _style_axis(ax, "Box covers the middle 50% of values")
    ax.annotate(f"Q1: {q1:,.2f}", xy=(q1, 0.1), fontsize=11, color="#0f172a")
    ax.annotate(f"Median: {median_v:,.2f}", xy=(median_v, 0.0), fontsize=11, color="#0f172a")
    ax.annotate(f"Q3: {q3:,.2f}", xy=(q3, -0.1), fontsize=11, color="#0f172a")

    outliers = values[(values < lower) | (values > upper)]
    if len(outliers) <= 5:
        for value in outliers:
            ax.text(float(value), 0.18, f"{float(value):,.2f}", fontsize=10, color="#7f1d1d")
    fig.tight_layout()
    fig.savefig(box_path)
    plt.close()

    return {
        "chart_histogram_url": f"/static/{upload_id}/{hist_path.name}",
        "chart_boxplot_url": f"/static/{upload_id}/{box_path.name}",
    }


def _save_categorical_chart(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, str]:
    chart_base = _safe_chart_name(column)
    out_dir = output_dir(upload_id)
    bar_path = out_dir / f"{chart_base}_bar.png"

    counts = dataframe[column].astype(str).value_counts(dropna=False).head(10)
    percentages = (counts / max(int(counts.sum()), 1)) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#cbd5e1"] * len(counts)
    if len(colors):
        colors[0] = "#2563eb"
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_title(f"Most common values in {column}", fontsize=14)
    _style_axis(ax, "Taller bars mean the value appears more often")
    ax.tick_params(axis="x", rotation=30)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{percentages.iloc[idx]:.1f}%", ha="center", va="bottom", fontsize=11)
    if len(percentages) and percentages.iloc[0] > 60:
        ax.annotate("One category dominates", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=11, color="#7c2d12")
    fig.tight_layout()
    fig.savefig(bar_path)
    plt.close()

    return {"chart_bar_url": f"/static/{upload_id}/{bar_path.name}"}


def _save_datetime_chart(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, str]:
    chart_base = _safe_chart_name(column)
    out_dir = output_dir(upload_id)
    line_path = out_dir / f"{chart_base}_line.png"

    dt_series = pd.to_datetime(dataframe[column], errors="coerce").dropna().sort_values()
    if dt_series.empty:
        return {"chart_line_url": ""}

    span_days = (dt_series.max() - dt_series.min()).days
    if span_days <= 90:
        freq = "D"
        period = "daily"
    elif span_days <= 720:
        freq = "W"
        period = "weekly"
    elif span_days <= 3650:
        freq = "M"
        period = "monthly"
    else:
        freq = "Y"
        period = "yearly"

    trend = dt_series.to_frame(name=column).set_index(column).resample(freq).size()
    full_index = pd.date_range(start=trend.index.min(), end=trend.index.max(), freq=freq)
    trend = trend.reindex(full_index, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend.index, trend.values, color="#0f766e", linewidth=2)
    ax.set_title(f"How {column} changes over time", fontsize=14)
    _style_axis(ax, "Look for rising, falling, and missing periods")

    if len(trend) > 1:
        x_vals = np.arange(len(trend))
        coeff = np.polyfit(x_vals, trend.values, 1)
        trend_line = np.poly1d(coeff)(x_vals)
        ax.plot(trend.index, trend_line, linestyle="--", color="orange", linewidth=2)

    high_idx = int(np.argmax(trend.values))
    low_idx = int(np.argmin(trend.values))
    ax.annotate(f"High: {trend.values[high_idx]}", (trend.index[high_idx], trend.values[high_idx]), fontsize=11)
    ax.annotate(f"Low: {trend.values[low_idx]}", (trend.index[low_idx], trend.values[low_idx]), fontsize=11)

    gaps = trend[trend == 0]
    for dt in gaps.index:
        ax.axvspan(dt, dt, color="#e5e7eb", alpha=0.4)
    if not gaps.empty:
        ax.annotate("Data gap", xy=(0.85, 0.9), xycoords="axes fraction", fontsize=11, color="#334155")

    fig.tight_layout()
    fig.savefig(line_path)
    plt.close()

    return {"chart_line_url": f"/static/{upload_id}/{line_path.name}", "period": period}


def _numeric_stats(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    series = pd.to_numeric(dataframe[column], errors="coerce")
    clean = series.dropna()
    shape_info = _classify_distribution_shape(clean)

    q1 = float(clean.quantile(0.25)) if not clean.empty else None
    q2 = float(clean.quantile(0.50)) if not clean.empty else None
    q3 = float(clean.quantile(0.75)) if not clean.empty else None
    iqr = float(q3 - q1) if q1 is not None and q3 is not None else None

    if clean.empty or iqr is None:
        outliers_iqr = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_iqr = int(((clean < lower) | (clean > upper)).sum())

    zscores = np.abs(zscore(clean, nan_policy="omit")) if len(clean) > 2 else np.array([])
    outliers_z = int((zscores > 3).sum()) if zscores.size else 0

    charts = _save_hist_kde_box(upload_id, dataframe, column)

    return {
        "column": column,
        "inferred_type": "numeric",
        "count": int(clean.count()),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "mean": float(clean.mean()) if not clean.empty else None,
        "median": float(clean.median()) if not clean.empty else None,
        "mode": float(clean.mode().iloc[0]) if not clean.mode().empty else None,
        "std": float(clean.std()) if not clean.empty else None,
        "variance": float(clean.var()) if not clean.empty else None,
        "min": float(clean.min()) if not clean.empty else None,
        "max": float(clean.max()) if not clean.empty else None,
        "range": float(clean.max() - clean.min()) if not clean.empty else None,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": iqr,
        "skewness": float(clean.skew()) if not clean.empty else None,
        "kurtosis": float(clean.kurtosis()) if not clean.empty else None,
        "outliers_iqr_count": outliers_iqr,
        "outliers_zscore_count": outliers_z,
        "distribution_shape": shape_info,
        **charts,
    }


def _categorical_stats(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    series = dataframe[column].astype("string")
    non_null = series.dropna()
    freq = non_null.value_counts(dropna=False)
    total = int(non_null.count())

    freq_table = [
        {
            "value": str(index),
            "count": int(count),
            "pct": round(float((count / max(total, 1)) * 100), 2),
        }
        for index, count in freq.items()
    ]

    charts = _save_categorical_chart(upload_id, dataframe, column)

    return {
        "column": column,
        "inferred_type": "categorical",
        "count": total,
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "mode": str(non_null.mode().iloc[0]) if not non_null.mode().empty else None,
        "cardinality": int(non_null.nunique()),
        "unique_count": int(non_null.nunique()),
        "top_10": freq_table[:10],
        "frequency_table": freq_table,
        **charts,
    }


def _datetime_stats(upload_id: str, dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    series = pd.to_datetime(dataframe[column], errors="coerce")
    non_null = series.dropna().sort_values()

    if non_null.empty:
        return {
            "column": column,
            "inferred_type": "datetime",
            "count": 0,
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "min_date": None,
            "max_date": None,
            "gap_count": 0,
            "observations_per_period": [],
            "chart_line_url": "",
        }

    charts = _save_datetime_chart(upload_id, dataframe, column)
    period = charts.get("period", "daily")
    freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "yearly": "Y"}
    trend = non_null.to_frame(name=column).set_index(column).resample(freq_map[period]).size()
    gaps = int((trend == 0).sum())

    return {
        "column": column,
        "inferred_type": "datetime",
        "count": int(non_null.count()),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "min_date": str(non_null.min()),
        "max_date": str(non_null.max()),
        "gap_count": gaps,
        "period": period,
        "observations_per_period": [
            {"period": str(index), "count": int(value)} for index, value in trend.items()
        ],
        "chart_line_url": charts.get("chart_line_url", ""),
    }


def _quick_column_stats(dataframe: pd.DataFrame, column: str, inferred_type: str) -> dict[str, Any]:
    """Compute lightweight stats without generating any chart images."""
    series = dataframe[column]
    missing_count = int(series.isna().sum())
    missing_pct = round(float(series.isna().mean() * 100), 2)

    base: dict[str, Any] = {
        "column": column,
        "inferred_type": inferred_type,
        "count": int(series.dropna().count()),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }

    if inferred_type == "numeric":
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if not clean.empty:
            shape_info = _classify_distribution_shape(clean)
            q1, q2, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.50)), float(clean.quantile(0.75))
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers_iqr = int(((clean < lower) | (clean > upper)).sum())
            base.update({
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "mode": float(clean.mode().iloc[0]) if not clean.mode().empty else None,
                "std": float(clean.std()),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "range": float(clean.max() - clean.min()),
                "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
                "skewness": float(clean.skew()),
                "kurtosis": float(clean.kurtosis()),
                "outliers_iqr_count": outliers_iqr,
                "distribution_shape": shape_info,
            })
    elif inferred_type == "datetime":
        dt = pd.to_datetime(series, errors="coerce").dropna().sort_values()
        if not dt.empty:
            base.update({
                "min_date": str(dt.min()),
                "max_date": str(dt.max()),
                "gap_count": 0,
            })
    else:
        non_null = series.dropna().astype(str)
        base.update({
            "mode": str(non_null.mode().iloc[0]) if not non_null.mode().empty else None,
            "cardinality": int(non_null.nunique()),
            "unique_count": int(non_null.nunique()),
        })

    return base


def _compute_dataset_trends(dataframe: pd.DataFrame, inferred_types: dict[str, str]) -> dict[str, Any]:
    """Compute high-level dataset trends: correlations, top numeric distributions, category dominance."""
    trends: dict[str, Any] = {}

    # Numeric correlations (top pairs)
    numeric_cols = [c for c, t in inferred_types.items() if t == "numeric"]
    if len(numeric_cols) >= 2:
        numeric_df = dataframe[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric_df.corr(method="pearson")
        # Find top correlated pairs (excluding self-correlation)
        pairs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    pairs.append({"col_a": c1, "col_b": c2, "r": round(float(val), 3)})
        pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
        trends["top_correlations"] = pairs[:5]

    # Numeric distribution summaries
    if numeric_cols:
        summaries = []
        for col in numeric_cols[:10]:
            clean = pd.to_numeric(dataframe[col], errors="coerce").dropna()
            if clean.empty:
                continue
            summaries.append({
                "column": col,
                "mean": round(float(clean.mean()), 2),
                "median": round(float(clean.median()), 2),
                "std": round(float(clean.std()), 2),
                "skew": round(float(clean.skew()), 2),
            })
        trends["numeric_summaries"] = summaries

    # Category dominance
    cat_cols = [c for c, t in inferred_types.items() if t in {"categorical", "boolean"}]
    if cat_cols:
        dominance = []
        for col in cat_cols[:10]:
            counts = dataframe[col].value_counts(dropna=True)
            if counts.empty:
                continue
            top_val = str(counts.index[0])
            top_pct = round(float(counts.iloc[0] / counts.sum() * 100), 1)
            dominance.append({
                "column": col,
                "top_value": top_val,
                "top_pct": top_pct,
                "unique": int(dataframe[col].nunique()),
            })
        trends["category_dominance"] = dominance

    # Datetime range
    dt_cols = [c for c, t in inferred_types.items() if t == "datetime"]
    for col in dt_cols[:3]:
        dt = pd.to_datetime(dataframe[col], errors="coerce").dropna().sort_values()
        if dt.empty:
            continue
        trends.setdefault("time_ranges", []).append({
            "column": col,
            "from": str(dt.min()),
            "to": str(dt.max()),
            "span_days": int((dt.max() - dt.min()).days),
        })

    return trends


def compute_analysis(upload_id: str, sheet_name: str | None = None, mode: str = "auto", focus_columns: list[str] | None = None) -> dict[str, Any]:
    """
    Modes:
      - auto:    quick if >50K rows, full otherwise
      - quick:   summary stats + trends only, no per-column charts (fast)
      - focused: full analysis only on focus_columns, quick stats for the rest
      - full:    current behaviour — every column gets charts
    """
    logger.info("[%s] Loading raw data (mode=%s)...", upload_id[:8], mode)
    dataframe = _load_raw_dataframe(upload_id, sheet_name=sheet_name)
    row_count = len(dataframe)
    col_count = len(dataframe.columns)
    logger.info("[%s] Loaded %d rows × %d cols", upload_id[:8], row_count, col_count)

    # Resolve 'auto' mode
    if mode == "auto":
        mode = "quick" if row_count > 50_000 else "full"
        logger.info("[%s] Auto-resolved mode → %s", upload_id[:8], mode)

    cleaned_df, cleaning_meta = _coerce_clean_dataframe(dataframe)
    inferred_types = {column: _infer_column_type(cleaned_df[column]) for column in cleaned_df.columns}
    correction = _smart_type_correction(cleaned_df, inferred_types)
    final_types: dict[str, str] = correction["final_types"]

    # Keep quality flags computed on final types where relevant
    flags = _quality_flags(cleaned_df, final_types)
    summary = _dataset_summary(cleaned_df, final_types, flags)
    logger.info("[%s] Quality score: %.1f", upload_id[:8], summary["quality_score"])

    # Decide which columns get charts vs quick stats
    focus_set = set(focus_columns) if focus_columns else set()
    excluded_from_stats = {item["column"] for item in correction["excluded_columns"]}
    if mode == "full":
        chart_columns = set(cleaned_df.columns) - excluded_from_stats
    elif mode == "focused":
        chart_columns = (focus_set & set(cleaned_df.columns)) - excluded_from_stats
    else:  # quick
        chart_columns = set()

    # For chart generation, sample large datasets to avoid matplotlib slowness
    if len(cleaned_df) > 50_000 and chart_columns:
        chart_df = cleaned_df.sample(n=50_000, random_state=42)
        logger.info("[%s] Sampling 50K rows for chart generation", upload_id[:8])
    else:
        chart_df = cleaned_df

    # Sample for quick stats computation (aggregates don't need all rows)
    if len(cleaned_df) > 100_000:
        stats_df = cleaned_df.sample(n=100_000, random_state=42)
        logger.info("[%s] Sampling 100K rows for stats computation", upload_id[:8])
    else:
        stats_df = cleaned_df

    column_stats: dict[str, Any] = {}
    total = len(final_types)
    for idx, (column, inferred_type) in enumerate(final_types.items(), 1):
        should_chart = column in chart_columns
        label = "full" if should_chart else "quick"
        logger.info("[%s] Processing column %d/%d: %s (%s, %s)", upload_id[:8], idx, total, column, inferred_type, label)

        if inferred_type in {"id", "constant", "free_text"}:
            series = cleaned_df[column]
            column_stats[column] = {
                "column": column,
                "inferred_type": inferred_type,
                "count": int(series.dropna().count()),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "excluded_from_statistical_analysis": True,
            }
            continue

        if should_chart:
            # Full analysis with charts
            if inferred_type == "numeric":
                column_stats[column] = _numeric_stats(upload_id, chart_df, column)
            elif inferred_type in {"categorical", "boolean", "id", "free_text"}:
                column_stats[column] = _categorical_stats(upload_id, chart_df, column)
            elif inferred_type == "datetime":
                column_stats[column] = _datetime_stats(upload_id, chart_df, column)
            else:
                column_stats[column] = _categorical_stats(upload_id, chart_df, column)
        else:
            # Quick stats — no chart generation
            column_stats[column] = _quick_column_stats(stats_df, column, inferred_type)

    # Use full dataframe for exact counts (missing, count, etc.)
    for column, inferred_type in final_types.items():
        series = cleaned_df[column]
        column_stats[column]["count"] = int(series.dropna().count())
        column_stats[column]["missing_count"] = int(series.isna().sum())
        column_stats[column]["missing_pct"] = round(float(series.isna().mean() * 100), 2)

    # Universal derived findings for every dataset (pre-AI)
    summary["trends"] = _compute_dataset_trends(cleaned_df, final_types)
    correlation_findings = _correlation_analysis(cleaned_df, final_types)
    group_differences = _group_difference_analysis(cleaned_df, final_types)
    outlier_characterization = _outlier_characterisation(cleaned_df, final_types)
    dataset_checks = _dataset_level_checks(cleaned_df, final_types)

    non_descriptive_names = sum(
        1 for col in cleaned_df.columns if col.lower() in {"x", "a", "b"} or re.match(r"^(col\d+|[a-z]?\d+)$", col.lower())
    )
    name_quality_flag = (
        "Column names are not descriptive — interpretations are based purely on value patterns, not column names."
        if non_descriptive_names >= max(2, len(cleaned_df.columns) // 2)
        else None
    )

    cleaned_csv_path = output_dir(upload_id) / "cleaned.csv"
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    row_count_int = int(summary["rows"])
    result = {
        "summary": summary,
        "analysis_mode": mode,
        "focus_columns": list(chart_columns),
        "pre_analysis": {
            "smart_type_correction": correction,
            "correlation_analysis": correlation_findings,
            "group_difference_analysis": group_differences,
            "outlier_characterisation": outlier_characterization,
            "dataset_level_checks": dataset_checks,
            "name_quality_flag": name_quality_flag,
        },
        "columns": [
            {
                "name": column,
                "inferred_type": final_types[column],
                "health": _compute_column_health(column_stats[column], row_count_int),
                "quality_flags": {
                    "missing_pct": flags["missing"][column]["pct"],
                    "missing_warn": flags["missing"][column]["warn"],
                    "constant": column in flags["constant_columns"],
                    "high_cardinality": column in flags["high_cardinality_columns"],
                    "potential_id": column in flags["potential_id_columns"],
                },
            }
            for column in cleaned_df.columns
        ],
        "column_stats": column_stats,
        "quality_flags": flags,
        "cleaning": cleaning_meta,
        "cleaned_csv_path": f"/static/{upload_id}/cleaned.csv",
    }

    for column in result["columns"]:
        col_name = column["name"]
        result["column_stats"][col_name]["health"] = column["health"]

    write_json(analysis_path(upload_id), result)
    logger.info("[%s] Analysis complete (mode=%s) — written to disk", upload_id[:8], mode)
    return result


def get_analysis(upload_id: str) -> dict[str, Any]:
    return read_json(analysis_path(upload_id), {})


def set_column_type_override(upload_id: str, column: str, new_type: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise ValueError("Analysis not found")

    for item in analysis.get("columns", []):
        if item["name"] == column:
            item["inferred_type"] = new_type
            break

    dataframe = _load_raw_dataframe(upload_id, sheet_name=read_json(metadata_path(upload_id), {}).get("active_sheet"))
    cleaned_df, _ = _coerce_clean_dataframe(dataframe)

    if new_type == "numeric":
        analysis["column_stats"][column] = _numeric_stats(upload_id, cleaned_df, column)
    elif new_type == "datetime":
        analysis["column_stats"][column] = _datetime_stats(upload_id, cleaned_df, column)
    else:
        analysis["column_stats"][column] = _categorical_stats(upload_id, cleaned_df, column)

    write_json(analysis_path(upload_id), analysis)
    return analysis["column_stats"][column]
