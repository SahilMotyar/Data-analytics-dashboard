from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from app.core.config import settings
from app.core.storage import read_json, write_json


def _fallback_column_summary(stats: dict[str, Any]) -> dict[str, str]:
    inferred_type = stats.get("inferred_type")
    missing = stats.get("missing_pct", 0)
    if inferred_type == "numeric":
        return {
            "what_does_this_look_like": (
                f"Most values sit around {stats.get('median')} while the average is {stats.get('mean')}. "
                f"Values span from {stats.get('min')} to {stats.get('max')} with {missing}% missing entries."
            ),
            "anything_unusual": (
                f"There are {stats.get('outliers_iqr_count', 0)} values that sit far from the typical range."
            ),
            "what_should_i_do": "Review outliers before using averages for business decisions.",
        }
    if inferred_type == "datetime":
        return {
            "what_does_this_look_like": (
                f"Dates run from {stats.get('min_date')} to {stats.get('max_date')} with {missing}% missing values."
            ),
            "anything_unusual": f"Detected {stats.get('gap_count', 0)} time gaps in the selected period.",
            "what_should_i_do": "Investigate missing periods before building trend-based reports.",
        }
    return {
        "what_does_this_look_like": (
            f"This column is mostly made of category labels. The most frequent value is {stats.get('mode')} with {missing}% missing values."
        ),
        "anything_unusual": (
            f"It has {stats.get('cardinality')} unique values, which may be high depending on business context."
        ),
        "what_should_i_do": "Standardize category names and merge near-duplicate labels before deeper analysis.",
    }


def _fallback_dataset_summary(analysis: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    return (
        f"The dataset has {summary.get('rows')} rows and {summary.get('columns')} columns with memory usage "
        f"around {summary.get('memory_mb')} MB. The overall quality score is {summary.get('quality_score')} out of 100. "
        "Main issues include missing data and potential high-cardinality or identifier-like columns."
    )


def _fallback_key_findings(analysis: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    summary = analysis.get("summary", {})
    columns = analysis.get("columns", [])
    warnings = []
    for column in columns:
        health = column.get("health", {}).get("overall", {})
        if health.get("status") in {"warning", "critical"}:
            warnings.append(f"{column.get('name')} needs attention: {health.get('label')}")

    top_findings = [
        f"The dataset contains {summary.get('rows')} rows across {summary.get('columns')} columns.",
        f"Overall quality score is {summary.get('quality_score')} out of 100.",
        "A small set of columns appears to need cleanup before reporting.",
    ]

    return {
        "whats_in_this_data": (
            f"{metadata.get('file_name', 'This file')} includes operational records with mixed data types. "
            "You can use it for trend and category analysis after quick cleanup checks."
        ),
        "top_findings": top_findings,
        "watch_out_for": warnings[:2],
        "suggested_next_step": "Review the red and yellow health columns first, then rerun summaries.",
    }


def _cache_path(file_hash: str) -> Path:
    cache_dir = settings.storage_dir / "ai_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{file_hash}.json"


def _safe_json_parse(text: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return default
    return default


def _extract_text(response: Any) -> str:
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def _build_column_profile(column_name: str, stats: dict[str, Any]) -> str:
    if stats.get("inferred_type") == "numeric":
        return (
            f"{column_name} | type=numeric | mean={stats.get('mean')} | median={stats.get('median')} | "
            f"missing={stats.get('missing_pct')}% | outliers={stats.get('outliers_iqr_count')} | skewness={stats.get('skewness')}"
        )
    if stats.get("inferred_type") == "categorical":
        top = stats.get("top_10", [])[:3]
        return (
            f"{column_name} | type=categorical | mode={stats.get('mode')} | missing={stats.get('missing_pct')}% | "
            f"top_categories={top}"
        )
    return (
        f"{column_name} | type={stats.get('inferred_type')} | missing={stats.get('missing_pct')}%"
    )


def _computed_facts_block(analysis: dict[str, Any], metadata: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    columns = analysis.get("columns", [])
    col_stats = analysis.get("column_stats", {})
    pre = analysis.get("pre_analysis", {})

    correction = pre.get("smart_type_correction", {})
    reclass = correction.get("reclassifications", [])
    reclass_map = {item.get("column"): item for item in reclass}

    column_lines = []
    for item in columns:
        col = item.get("name")
        typ = item.get("inferred_type")
        stats = col_stats.get(col, {})
        rec = reclass_map.get(col)
        rec_text = f" | reclassified: {rec.get('from')}→{rec.get('to')} ({rec.get('reason')})" if rec else ""
        if typ == "numeric":
            shape = (stats.get("distribution_shape") or {}).get("shape")
            column_lines.append(
                f"\"{col}\" | type={typ}{rec_text} | mean={stats.get('mean')}, median={stats.get('median')}, std={stats.get('std')}, range=[{stats.get('min')},{stats.get('max')}], shape={shape}, outliers={stats.get('outliers_iqr_count')}, missing={stats.get('missing_pct')}%"
            )
        elif typ in {"categorical", "boolean"}:
            top = (stats.get("top_10") or [{}])[0] if stats.get("top_10") else {}
            column_lines.append(
                f"\"{col}\" | type={typ}{rec_text} | unique={stats.get('unique_count') or stats.get('cardinality')}, top={top.get('value')}({top.get('pct')}%), missing={stats.get('missing_pct')}%"
            )
        elif typ == "datetime":
            column_lines.append(
                f"\"{col}\" | type={typ}{rec_text} | range={stats.get('min_date')} to {stats.get('max_date')}, gaps={stats.get('gap_count')}, missing={stats.get('missing_pct')}%"
            )
        else:
            column_lines.append(f"\"{col}\" | type={typ}{rec_text} | missing={stats.get('missing_pct')}%")

    corr = pre.get("correlation_analysis", {})
    top_corr = corr.get("pairs", [])[:5]
    corr_lines = [
        f"{item.get('col_a')}↔{item.get('col_b')}: r={item.get('pearson_r')} ({item.get('strength')})"
        + (" [non-linear signal]" if item.get("non_linear_signal") else "")
        for item in top_corr
    ]
    neg_lines = [
        f"{item.get('col_a')}↔{item.get('col_b')}: r={item.get('pearson_r')}"
        for item in corr.get("notable_negative", [])
    ]
    red_lines = [f"{item.get('col_a')}↔{item.get('col_b')} (r={item.get('pearson_r')})" for item in corr.get("redundant_pairs", [])]

    effects = pre.get("group_difference_analysis", {}).get("strongest_by_category", [])
    effect_lines = [
        f"{item.get('categorical_column')} explains {item.get('numeric_column')} with effect={item.get('effect_size')} and means={item.get('group_means')}"
        for item in effects
    ]

    outliers = pre.get("outlier_characterisation", {}).get("multi_column_anomalies", [])
    outlier_lines = [f"row {item.get('row_index')} in columns {item.get('columns')}" for item in outliers]

    dataset_checks = pre.get("dataset_level_checks", {})
    sample_flags = dataset_checks.get("sample_size_flags", [])

    return (
        f"FILE: {metadata.get('file_name')}\n"
        f"ROWS: {summary.get('rows')} | COLUMNS: {summary.get('columns')} | MEMORY: {summary.get('memory_mb')}MB\n\n"
        f"COLUMN SUMMARY (post-reclassification):\n" + "\n".join(column_lines) + "\n\n"
        f"CROSS-COLUMN FINDINGS:\n"
        f"Top correlations:\n" + ("\n".join(corr_lines) if corr_lines else "none") + "\n\n"
        f"Notable negatives:\n" + ("\n".join(neg_lines) if neg_lines else "none") + "\n\n"
        f"Redundant pairs (|r|>0.95):\n" + ("\n".join(red_lines) if red_lines else "none") + "\n\n"
        f"Group differences:\n" + ("\n".join(effect_lines) if effect_lines else "none") + "\n\n"
        f"Reclassifications made:\n" + (json.dumps(reclass, default=str) if reclass else "none") + "\n\n"
        f"Bimodal/multimodal columns:\n" + (
            "\n".join(
                f"{name}: {(stats.get('distribution_shape') or {}).get('shape')}"
                for name, stats in col_stats.items()
                if (stats.get("distribution_shape") or {}).get("shape") in {"bimodal", "multimodal"}
            )
            or "none"
        )
        + "\n\n"
        f"Multi-column outlier rows:\n" + ("\n".join(outlier_lines) if outlier_lines else "none") + "\n\n"
        f"Sample size flags:\n" + ("\n".join(sample_flags) if sample_flags else "none")
    )


def _key_findings_prompt(analysis: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are a statistician writing a 4-sentence briefing about a dataset. "
        "Every claim must be grounded in the computed facts provided by the user. "
        "Sentence 1: infer what the data likely measures from column names/value patterns; if names are ambiguous, say so. "
        "Sentence 2: single most statistically interesting cross-column finding with actual numbers. "
        "Sentence 3: biggest quality/interpretation risk with numbers. "
        "Sentence 4: most useful concrete next action tied to specific columns/findings. "
        "Never invent missing facts. Do not include filler or generic statements."
    )
    user_prompt = (
        _computed_facts_block(analysis, metadata)
        + "\n\nReturn strict JSON with keys: whats_in_this_data (string), top_findings (array len 3), watch_out_for (array len up to 2), suggested_next_step (string)."
    )
    return system_prompt, user_prompt


def _column_prompt(column_name: str, stats: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are explaining one data column to a mixed audience. "
        "Use only supplied computed values. No fabricated numbers."
    )
    if stats.get("inferred_type") == "numeric":
        user_prompt = (
            f"Column: {column_name}\n"
            f"DISTRIBUTION: shape={stats.get('distribution_shape')}, mean={stats.get('mean')}, median={stats.get('median')}, mode={stats.get('mode')}, std={stats.get('std')}, "
            f"range={stats.get('min')} to {stats.get('max')}, skewness={stats.get('skewness')}, kurtosis={stats.get('kurtosis')}, outliers_iqr={stats.get('outliers_iqr_count')}, missing={stats.get('missing_pct')}%\n\n"
            "Write 3 short paragraphs: (1) shape/spread (2) relationships or independence (3) watch-outs. Then return JSON keys: what_does_this_look_like, anything_unusual, what_should_i_do."
        )
    else:
        top = stats.get("top_10", [])[:5]
        user_prompt = (
            f"Column: {column_name}\n"
            f"Top categories: {top}\n"
            f"Total unique values: {stats.get('cardinality')}\n"
            f"Missing: {stats.get('missing_pct')}%\n\n"
            "Write 2 short paragraphs: (1) composition/balance (2) what this column explains. Then return JSON keys: what_does_this_look_like, anything_unusual, what_should_i_do."
        )
    return system_prompt, user_prompt


def _call_claude_json(client: Anthropic, system_prompt: str, user_prompt: str, default: dict[str, Any]) -> dict[str, Any]:
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1200,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _safe_json_parse(_extract_text(response), default)


def generate_ai_summaries(analysis: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    file_hash = metadata.get("file_hash") or sha256(f"{metadata.get('file_name')}|{analysis.get('summary', {}).get('rows')}".encode("utf-8")).hexdigest()
    cached = read_json(_cache_path(file_hash), None)
    if cached:
        analysis["key_findings"] = cached.get("key_findings", _fallback_key_findings(analysis, metadata))
        analysis["dataset_overview"] = cached.get("dataset_overview", _fallback_dataset_summary(analysis))
        for column_name, stats in analysis.get("column_stats", {}).items():
            analysis["column_stats"][column_name]["ai_summary"] = cached.get("column_summaries", {}).get(
                column_name, _fallback_column_summary(stats)
            )
        return analysis

    if not settings.anthropic_api_key:
        analysis["key_findings"] = _fallback_key_findings(analysis, metadata)
        analysis["dataset_overview"] = _fallback_dataset_summary(analysis)
        for column_name, stats in analysis.get("column_stats", {}).items():
            fallback = _fallback_column_summary(stats)
            analysis["column_stats"][column_name]["ai_summary"] = fallback
            analysis["column_stats"][column_name]["insight_summary"] = " ".join(fallback.values())
        write_json(
            _cache_path(file_hash),
            {
                "key_findings": analysis["key_findings"],
                "dataset_overview": analysis["dataset_overview"],
                "column_summaries": {name: analysis["column_stats"][name]["ai_summary"] for name in analysis.get("column_stats", {})},
            },
        )
        return analysis

    client = Anthropic(api_key=settings.anthropic_api_key)
    key_system, key_user = _key_findings_prompt(analysis, metadata)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures: dict[str, Any] = {}
        futures["__key_findings__"] = pool.submit(
            _call_claude_json,
            client,
            key_system,
            key_user,
            _fallback_key_findings(analysis, metadata),
        )

        for column_name, stats in analysis.get("column_stats", {}).items():
            system_prompt, user_prompt = _column_prompt(column_name, stats)
            futures[column_name] = pool.submit(
                _call_claude_json,
                client,
                system_prompt,
                user_prompt,
                _fallback_column_summary(stats),
            )

        key_findings = _fallback_key_findings(analysis, metadata)
        column_summaries: dict[str, Any] = {}
        for name, future in futures.items():
            result = future.result()
            if name == "__key_findings__":
                key_findings = result
            else:
                column_summaries[name] = result

    analysis["key_findings"] = key_findings
    analysis["dataset_overview"] = analysis["key_findings"].get("whats_in_this_data", _fallback_dataset_summary(analysis))
    for column_name, stats in analysis.get("column_stats", {}).items():
        ai_summary = column_summaries.get(column_name, _fallback_column_summary(stats))
        analysis["column_stats"][column_name]["ai_summary"] = ai_summary
        analysis["column_stats"][column_name]["insight_summary"] = " ".join(ai_summary.values())

    write_json(
        _cache_path(file_hash),
        {
            "key_findings": analysis["key_findings"],
            "dataset_overview": analysis["dataset_overview"],
            "column_summaries": column_summaries,
        },
    )
    return analysis


def generate_chat_answer(analysis: dict[str, Any], metadata: dict[str, Any], message: str, history: list[dict[str, str]]) -> str:
    if not settings.anthropic_api_key:
        return "I can help with this dataset based on computed stats. Start with columns marked red or yellow, then review missing data and outliers before modeling."

    client = Anthropic(api_key=settings.anthropic_api_key)
    history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in history[-10:])
    prompt = (
        "DATASET FACTS:\n"
        f"{_computed_facts_block(analysis, metadata)}\n\n"
        f"KEY BRIEFING ALREADY GENERATED:\n{json.dumps(analysis.get('key_findings', {}), default=str)}\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\n"
        f"USER QUESTION:\n{message}"
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=700,
        temperature=0,
        system=(
            "You are a data analyst assistant. Answer using only computed facts provided. "
            "Never invent numbers. If a requested fact is absent, say: 'I don't have that information from this dataset.' "
            "Use actual column names and values. 2-4 sentences for simple questions."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response).strip()
    return text or "I couldn't generate a response right now."
