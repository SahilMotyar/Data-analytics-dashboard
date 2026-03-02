"""
Nano Banana – Instant Automated Insights Engine v1.2
====================================================
Unified, stateless analysis pipeline.

Pipeline stages:
1.  read_file        → pandas DataFrame
2.  detect_types     → column‑type map
3.  compute_quality  → A–F grade + readable summary
4.  column_profiles  → per‑column stats + narrative
5.  correlations     → strong pairs with text
6.  outlier_scan     → IQR‑based anomaly list
7.  pareto_check     → 80/20 analysis on categoricals
8.  build_charts     → Recharts‑ready JSON
9.  headline         → single "One Big Thing" sentence

Everything returns plain dicts / lists safe for JSON serialisation.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── constants ────────────────────────────────────────────────────────────
MAX_LINE_PTS = 300
MAX_SCATTER_PTS = 500
MAX_BAR_ITEMS = 15
MAX_CHARTS = 8
CORR_THRESHOLD = 0.7
OUTLIER_IQR_FACTOR = 1.5

GRADE_TABLE = [
    (97, "A+"), (93, "A"), (90, "A-"),
    (87, "B+"), (83, "B"), (80, "B-"),
    (77, "C+"), (73, "C"), (70, "C-"),
    (67, "D+"), (63, "D"), (60, "D-"),
    (0, "F"),
]


# ── helpers ──────────────────────────────────────────────────────────────
def _safe(val: Any) -> Any:
    """Convert numpy / pandas scalars to JSON‑safe Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if (np.isnan(val) or np.isinf(val)) else round(float(val), 4)
    if isinstance(val, np.ndarray):
        return [_safe(v) for v in val]
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _fmt(val: float | int, precision: int = 1) -> str:
    """Human‑friendly number formatting."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, np.integer)):
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:,.{precision}f}M"
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:,.{precision}f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:,.{precision}f}K"
        return f"{val:,.{precision}f}"
    return str(val)


def _pct(part: float, whole: float) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def _grade(score: float) -> str:
    for threshold, label in GRADE_TABLE:
        if score >= threshold:
            return label
    return "F"


# ── 1. File reading ─────────────────────────────────────────────────────
def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        # Try common encodings for CSV
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(
                "We couldn't read that file – please make sure it's a valid CSV or Excel file."
            )

    if df.empty or len(df) == 0:
        raise ValueError("The uploaded file appears to be empty.")

    # Auto‑parse potential date columns
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, format="mixed", dayfirst=False)
            if parsed.notna().sum() >= len(sample) * 0.7:
                df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
        except Exception:
            pass

    return df


# ── 2. Type detection ───────────────────────────────────────────────────
def detect_types(df: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numerical"
        else:
            nunique = df[col].nunique()
            if nunique <= 25 or (nunique / max(len(df), 1) < 0.05):
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


# ── 3. Quality scoring ──────────────────────────────────────────────────
def compute_quality(df: pd.DataFrame, types: dict[str, str]) -> dict:
    total_cells = df.shape[0] * df.shape[1]

    # Missing values
    missing_per_col: dict[str, int] = {}
    total_missing = 0
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            missing_per_col[col] = n
            total_missing += n

    missing_pct = _pct(total_missing, total_cells)

    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_pct = _pct(dup_count, len(df))

    # Outliers (numerical cols only)
    outlier_cols: dict[str, int] = {}
    total_outliers = 0
    for col, t in types.items():
        if t != "numerical":
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            outlier_cols[col] = n_out
            total_outliers += n_out

    outlier_pct = _pct(total_outliers, total_cells)

    # Score (0‑100)
    score = max(0, 100 - missing_pct * 0.6 - dup_pct * 0.25 - outlier_pct * 0.15)
    score = round(score, 1)

    # Readable summary
    parts: list[str] = []
    if total_missing:
        worst = max(missing_per_col, key=missing_per_col.get) if missing_per_col else ""
        parts.append(
            f"We found {total_missing:,} missing value{'s' if total_missing > 1 else ''}"
            + (f", mostly in the '{worst}' column" if worst else "")
        )
    if dup_count:
        parts.append(f"{dup_count:,} duplicate row{'s' if dup_count > 1 else ''}")
    if total_outliers:
        parts.append(f"{total_outliers:,} statistical outlier{'s' if total_outliers > 1 else ''}")

    if not parts:
        summary = f"Your data is {score:.0f}% clean. No critical errors found."
    else:
        summary = f"Your data is {score:.0f}% clean. {'. '.join(parts)}."

    return {
        "score": score,
        "grade": _grade(score),
        "summary": summary,
        "missing": {"total": total_missing, "pct": missing_pct, "columns": missing_per_col},
        "duplicates": {"count": dup_count, "pct": dup_pct},
        "outliers": {"total": total_outliers, "pct": outlier_pct, "columns": outlier_cols},
    }


# ── 4. Column profiles ──────────────────────────────────────────────────
def column_profiles(df: pd.DataFrame, types: dict[str, str]) -> list[dict]:
    profiles: list[dict] = []

    for col, ctype in types.items():
        entry: dict[str, Any] = {"name": col, "type": ctype, "non_null": int(df[col].notna().sum())}

        if ctype == "numerical":
            s = df[col].dropna()
            if len(s) == 0:
                entry["stats"] = {}
                entry["narrative"] = f"'{col}' has no valid values."
                profiles.append(entry)
                continue

            mean_v = _safe(s.mean())
            median_v = _safe(s.median())
            std_v = _safe(s.std())
            min_v = _safe(s.min())
            max_v = _safe(s.max())
            q1 = _safe(s.quantile(0.25))
            q3 = _safe(s.quantile(0.75))
            skew_v = _safe(s.skew())
            kurt_v = _safe(s.kurtosis())

            entry["stats"] = {
                "mean": mean_v, "median": median_v, "std": std_v,
                "min": min_v, "max": max_v, "q1": q1, "q3": q3,
                "skew": skew_v, "kurtosis": kurt_v,
            }

            # Distribution shape for histogram
            try:
                hist_counts, bin_edges = np.histogram(s, bins=min(30, max(10, len(s) // 20)))
                entry["histogram"] = [
                    {"bin": _safe(round((bin_edges[i] + bin_edges[i + 1]) / 2, 2)),
                     "count": int(hist_counts[i])}
                    for i in range(len(hist_counts))
                ]
            except Exception:
                entry["histogram"] = []

            # Narrative
            direction = ""
            if mean_v and median_v:
                ratio = abs(mean_v - median_v) / max(abs(median_v), 1e-9)
                if ratio > 0.15:
                    direction = (
                        " The mean is noticeably higher than the median, suggesting a right‑skewed distribution."
                        if mean_v > median_v
                        else " The mean is below the median, suggesting a left‑skewed distribution."
                    )
            entry["narrative"] = (
                f"'{col}' averages {_fmt(mean_v)} (median {_fmt(median_v)}), "
                f"ranging from {_fmt(min_v)} to {_fmt(max_v)}.{direction}"
            )

        elif ctype == "categorical":
            vc = df[col].value_counts()
            top = vc.head(MAX_BAR_ITEMS)
            mode_val = vc.index[0] if len(vc) > 0 else "N/A"
            mode_pct = _pct(vc.iloc[0], len(df)) if len(vc) > 0 else 0

            entry["stats"] = {
                "unique": int(df[col].nunique()),
                "mode": str(mode_val),
                "mode_pct": mode_pct,
                "top_values": [{"value": str(k), "count": int(v)} for k, v in top.items()],
            }
            entry["narrative"] = (
                f"'{col}' has {int(df[col].nunique())} unique values. "
                f"'{mode_val}' is the most common, appearing {mode_pct}% of the time."
            )

        elif ctype == "datetime":
            s = df[col].dropna()
            if len(s) == 0:
                entry["stats"] = {}
                entry["narrative"] = f"'{col}' has no valid dates."
                profiles.append(entry)
                continue

            min_d = _safe(s.min())
            max_d = _safe(s.max())
            span_days = (s.max() - s.min()).days

            entry["stats"] = {"min": min_d, "max": max_d, "span_days": span_days}

            if span_days > 365:
                span_str = f"{span_days // 365} year{'s' if span_days // 365 > 1 else ''}"
            elif span_days > 30:
                span_str = f"{span_days // 30} month{'s' if span_days // 30 > 1 else ''}"
            else:
                span_str = f"{span_days} day{'s' if span_days != 1 else ''}"
            entry["narrative"] = f"'{col}' spans {span_str}, from {str(min_d)[:10]} to {str(max_d)[:10]}."

        else:
            entry["stats"] = {"unique": int(df[col].nunique())}
            entry["narrative"] = f"'{col}' is a free‑text field with {int(df[col].nunique())} unique entries."

        profiles.append(entry)

    return profiles


# ── 5. Correlation analysis ─────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame, types: dict[str, str]) -> list[dict]:
    num_cols = [c for c, t in types.items() if t == "numerical"]
    if len(num_cols) < 2:
        return []

    corr_matrix = df[num_cols].corr(method="pearson")
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if i >= j:
                continue
            r = corr_matrix.loc[a, b]
            if np.isnan(r):
                continue
            if abs(r) < CORR_THRESHOLD:
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))

            direction = "increases" if r > 0 else "decreases"
            strength = "Strong" if abs(r) > 0.85 else "Moderate"

            results.append({
                "col_a": a,
                "col_b": b,
                "value": round(float(r), 3),
                "strength": strength.lower(),
                "narrative": (
                    f"{strength} relationship detected: as '{a}' increases, "
                    f"'{b}' usually {direction} (r = {r:.2f})."
                ),
            })

    results.sort(key=lambda x: abs(x["value"]), reverse=True)
    return results


# ── 6. Outlier detection ────────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame, types: dict[str, str]) -> list[dict]:
    results: list[dict] = []

    for col, t in types.items():
        if t != "numerical":
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - OUTLIER_IQR_FACTOR * iqr
        upper = q3 + OUTLIER_IQR_FACTOR * iqr
        mask = (s < lower) | (s > upper)
        outliers = s[mask]

        if len(outliers) == 0:
            continue

        mean_v = s.mean()
        examples: list[dict] = []
        for val in outliers.head(5):
            ratio = abs(val / mean_v) if mean_v != 0 else 0
            direction = "higher" if val > mean_v else "lower"
            examples.append({
                "value": _safe(val),
                "explanation": f"Value {_fmt(val)} is {ratio:.1f}× the average ({direction} than expected).",
            })

        results.append({
            "column": col,
            "count": int(len(outliers)),
            "lower_fence": _safe(round(float(lower), 2)),
            "upper_fence": _safe(round(float(upper), 2)),
            "examples": examples,
            "narrative": (
                f"Found {len(outliers)} outlier{'s' if len(outliers) > 1 else ''} in '{col}'. "
                f"Values outside [{_fmt(lower)} – {_fmt(upper)}] are flagged."
            ),
        })

    return results


# ── 7. Pareto analysis ──────────────────────────────────────────────────
def compute_pareto(df: pd.DataFrame, types: dict[str, str]) -> list[dict]:
    cat_cols = [c for c, t in types.items() if t == "categorical"]
    num_cols = [c for c, t in types.items() if t == "numerical"]

    results: list[dict] = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=True).sort_values(ascending=False)
        if len(vc) < 3:
            continue

        cumsum = vc.cumsum()
        total = vc.sum()
        top_20_cutoff = max(1, math.ceil(len(vc) * 0.2))
        top_20_share = _pct(cumsum.iloc[min(top_20_cutoff - 1, len(cumsum) - 1)], total)

        applies = top_20_share >= 65  # relaxed from strict 80%

        # If a numerical column exists, try to compute value‑weighted Pareto
        value_col = None
        value_weighted_pct = None
        if num_cols:
            # pick first numerical col
            for nc in num_cols:
                grouped = df.groupby(col)[nc].sum().sort_values(ascending=False)
                gtotal = grouped.sum()
                if gtotal > 0:
                    gcum = grouped.cumsum()
                    g_top20 = max(1, math.ceil(len(grouped) * 0.2))
                    value_weighted_pct = _pct(
                        gcum.iloc[min(g_top20 - 1, len(gcum) - 1)], gtotal
                    )
                    if value_weighted_pct >= 65:
                        value_col = nc
                        applies = True
                        break

        if not applies:
            continue

        # Build Pareto data (for chart)
        pareto_data: list[dict] = []
        running = 0
        for val, cnt in vc.items():
            running += cnt
            pareto_data.append({
                "value": str(val),
                "count": int(cnt),
                "cumulative_pct": round(running / total * 100, 1),
            })

        narrative = f"Top {top_20_cutoff} of {len(vc)} '{col}' values account for {top_20_share}% of occurrences"
        if value_col and value_weighted_pct:
            narrative += f" and {value_weighted_pct}% of total '{value_col}'"
        narrative += ". The 80/20 rule applies here."

        results.append({
            "column": col,
            "applies": True,
            "top_20_count": top_20_cutoff,
            "top_20_share": top_20_share,
            "value_weighted_col": value_col,
            "value_weighted_pct": value_weighted_pct,
            "narrative": narrative,
            "data": pareto_data[:MAX_BAR_ITEMS],
        })

    return results


# ── 8. Chart generation ─────────────────────────────────────────────────
def _resample_series(dates: pd.Series, values: pd.Series, max_pts: int) -> pd.DataFrame:
    """Aggregate a time‑series to fit within max_pts."""
    tmp = pd.DataFrame({"date": dates, "value": values}).dropna(subset=["date", "value"])
    tmp = tmp.set_index("date").sort_index()

    if len(tmp) <= max_pts:
        return tmp.reset_index()

    span = (tmp.index.max() - tmp.index.min()).days
    if span > 365 * 2:
        rule = "ME"   # monthly
    elif span > 180:
        rule = "W"
    else:
        rule = "D"

    resampled = tmp.resample(rule).mean().dropna()
    if len(resampled) > max_pts:
        step = max(1, len(resampled) // max_pts)
        resampled = resampled.iloc[::step]

    return resampled.reset_index()


def build_charts(
    df: pd.DataFrame,
    types: dict[str, str],
    correlations: list[dict],
) -> list[dict]:
    charts: list[dict] = []
    dt_cols = [c for c, t in types.items() if t == "datetime"]
    num_cols = [c for c, t in types.items() if t == "numerical"]
    cat_cols = [c for c, t in types.items() if t == "categorical"]

    # ── Time + Number → Line chart ──
    for dc in dt_cols:
        for nc in num_cols:
            if len(charts) >= MAX_CHARTS:
                break
            series = _resample_series(df[dc], df[nc], MAX_LINE_PTS)
            if len(series) < 3:
                continue

            # Trend percentage
            first_val = series["value"].iloc[0]
            last_val = series["value"].iloc[-1]
            if first_val and first_val != 0:
                pct_change = round((last_val - first_val) / abs(first_val) * 100, 1)
                direction = "grown" if pct_change > 0 else "declined"
                subtitle = f"'{nc}' has {direction} {abs(pct_change)}% over this period."
            else:
                subtitle = f"Trend of '{nc}' over time."

            # Simple linear trendline
            x_num = np.arange(len(series))
            if len(x_num) >= 2:
                slope, intercept, *_ = sp_stats.linregress(x_num, series["value"].values)
                trend_vals = slope * x_num + intercept
            else:
                trend_vals = series["value"].values

            data = []
            for i, row in series.iterrows():
                d = row["date"]
                data.append({
                    "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "value": _safe(row["value"]),
                    "trend": _safe(round(float(trend_vals[i if isinstance(i, int) else 0]), 2)),
                })

            # Fix trend indexing after reset_index
            for idx, item in enumerate(data):
                if idx < len(trend_vals):
                    item["trend"] = _safe(round(float(trend_vals[idx]), 2))

            charts.append({
                "id": f"line_{dc}_{nc}",
                "type": "line",
                "title": f"{nc} Trend",
                "subtitle": subtitle,
                "data": data,
                "xKey": "date",
                "yKeys": ["value", "trend"],
            })
        if len(charts) >= MAX_CHARTS:
            break

    # ── Category + Number → Bar chart ──
    for cc in cat_cols:
        for nc in num_cols:
            if len(charts) >= MAX_CHARTS:
                break
            grouped = df.groupby(cc)[nc].sum().sort_values(ascending=False).head(MAX_BAR_ITEMS)
            if len(grouped) < 2:
                continue

            total = grouped.sum()
            top_name = grouped.index[0]
            top_pct = _pct(grouped.iloc[0], total)
            subtitle = f"'{top_name}' accounts for {top_pct}% of total '{nc}'."

            data = [{"name": str(k), "value": _safe(v)} for k, v in grouped.items()]

            charts.append({
                "id": f"bar_{cc}_{nc}",
                "type": "bar",
                "title": f"{nc} by {cc}",
                "subtitle": subtitle,
                "data": data,
                "xKey": "name",
                "yKey": "value",
            })
        if len(charts) >= MAX_CHARTS:
            break

    # ── Number + Number → Scatter (for correlated pairs) ──
    for corr in correlations[:3]:
        if len(charts) >= MAX_CHARTS:
            break
        a, b = corr["col_a"], corr["col_b"]
        subset = df[[a, b]].dropna()
        if len(subset) > MAX_SCATTER_PTS:
            subset = subset.sample(MAX_SCATTER_PTS, random_state=42)

        # Regression line
        slope, intercept, *_ = sp_stats.linregress(subset[a], subset[b])
        x_range = np.linspace(subset[a].min(), subset[a].max(), 50)
        reg_line = [{"x": _safe(round(float(x), 2)), "y": _safe(round(float(slope * x + intercept), 2))} for x in x_range]

        data = [{"x": _safe(row[a]), "y": _safe(row[b])} for _, row in subset.iterrows()]

        charts.append({
            "id": f"scatter_{a}_{b}",
            "type": "scatter",
            "title": f"{a} vs {b}",
            "subtitle": corr["narrative"],
            "data": data,
            "xLabel": a,
            "yLabel": b,
            "regression": reg_line,
        })

    # ── Category + Category → Heatmap ──
    if len(cat_cols) >= 2 and len(charts) < MAX_CHARTS:
        c1, c2 = cat_cols[0], cat_cols[1]
        ct = pd.crosstab(df[c1], df[c2])
        # Limit to top categories
        ct = ct.iloc[:12, :12]

        heat_data: list[dict] = []
        max_val = ct.values.max() if ct.values.size else 1
        for r in ct.index:
            for c in ct.columns:
                heat_data.append({
                    "x": str(c),
                    "y": str(r),
                    "value": int(ct.loc[r, c]),
                    "intensity": round(float(ct.loc[r, c]) / max(max_val, 1), 3),
                })

        top_cell = ct.stack().idxmax() if ct.values.size else ("", "")
        subtitle = f"'{top_cell[0]}' × '{top_cell[1]}' is the most common combination."

        charts.append({
            "id": f"heatmap_{c1}_{c2}",
            "type": "heatmap",
            "title": f"{c1} × {c2} Distribution",
            "subtitle": subtitle,
            "data": heat_data,
            "xLabels": [str(c) for c in ct.columns.tolist()],
            "yLabels": [str(r) for r in ct.index.tolist()],
        })

    return charts


# ── 9. Headline generation ──────────────────────────────────────────────
def generate_headline(
    quality: dict,
    correlations: list[dict],
    outliers: list[dict],
    pareto: list[dict],
    profiles: list[dict],
    charts: list[dict],
) -> str:
    """Pick the single most interesting finding and craft a headline."""
    candidates: list[tuple[float, str]] = []

    # Quality issues
    if quality["score"] < 80:
        candidates.append((
            90 - quality["score"],
            f"Data quality needs attention — only {quality['score']:.0f}% clean.",
        ))

    # Trend insight (from line charts)
    for ch in charts:
        if ch["type"] == "line" and ch["data"]:
            first = ch["data"][0].get("value")
            last = ch["data"][-1].get("value")
            if first and last and first != 0:
                pct = round((last - first) / abs(first) * 100, 1)
                col_name = ch["title"].replace(" Trend", "")
                if abs(pct) >= 5:
                    direction = "trending upwards" if pct > 0 else "trending downwards"
                    candidates.append((
                        abs(pct),
                        f"{col_name} is {direction}, with a {abs(pct)}% change over the period.",
                    ))

    # Correlation insight
    for c in correlations[:2]:
        candidates.append((
            abs(c["value"]) * 50,
            c["narrative"],
        ))

    # Anomaly insight
    for o in outliers[:2]:
        candidates.append((
            o["count"] * 5,
            f"Unusual values detected in '{o['column']}' — {o['count']} data point{'s' if o['count'] > 1 else ''} fall outside the expected range.",
        ))

    # Pareto insight
    for p in pareto[:1]:
        if p["top_20_share"] >= 75:
            candidates.append((
                p["top_20_share"] * 0.5,
                p["narrative"],
            ))

    if not candidates:
        return "Your data has been analyzed. Explore the insights below."

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Main pipeline ───────────────────────────────────────────────────────
def analyze(file_bytes: bytes, filename: str) -> dict:
    """Run the full zero‑config analysis and return JSON‑safe results."""
    df = read_file(file_bytes, filename)
    types = detect_types(df)
    quality = compute_quality(df, types)
    profiles = column_profiles(df, types)
    corrs = compute_correlations(df, types)
    outs = detect_outliers(df, types)
    par = compute_pareto(df, types)
    charts = build_charts(df, types, corrs)
    headline = generate_headline(quality, corrs, outs, par, profiles, charts)

    return {
        "filename": filename,
        "row_count": len(df),
        "col_count": len(df.columns),
        "headline": headline,
        "quality": quality,
        "columns": profiles,
        "correlations": corrs,
        "outliers": outs,
        "pareto": par,
        "charts": charts,
    }
