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
                f"Most values for this column sit comfortably around {stats.get('median')}. "
                f"The data ranges from {stats.get('min')} to {stats.get('max')}, "
                f"with {missing}% of entries missing."
            ),
            "anything_unusual": (
                f"There are {stats.get('outliers_iqr_count', 0)} unusual data points that fall well outside the typical range — worth a closer look."
            ),
            "what_should_i_do": "Check those unusual values before relying on averages for any decisions.",
        }
    if inferred_type == "datetime":
        return {
            "what_does_this_look_like": (
                f"This timeline runs from {stats.get('min_date')} to {stats.get('max_date')}, with {missing}% of dates missing."
            ),
            "anything_unusual": f"There are {stats.get('gap_count', 0)} gaps in the timeline where data seems to be missing.",
            "what_should_i_do": "Look into the gaps — missing time periods can distort any trend you try to read.",
        }
    return {
        "what_does_this_look_like": (
            f"This column contains category labels. The most common value is \"{stats.get('mode')}\" and {missing}% of entries are blank."
        ),
        "anything_unusual": (
            f"It has {stats.get('cardinality')} different values — if that feels high, some categories may need merging."
        ),
        "what_should_i_do": "Clean up category names and merge near-duplicates before drawing conclusions.",
    }


def _fallback_dataset_summary(analysis: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    return (
        f"You're looking at a dataset with {summary.get('rows')} rows and {summary.get('columns')} columns. "
        f"Overall data quality scores {summary.get('quality_score')} out of 100. "
        "The main things to watch for are missing entries and a few columns that may be IDs rather than useful data."
    )


def _fallback_key_findings(analysis: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    summary = analysis.get("summary", {})
    columns = analysis.get("columns", [])
    warnings = []
    for column in columns:
        health = column.get("health", {}).get("overall", {})
        if health.get("status") in {"warning", "critical"}:
            warnings.append(f"\"{column.get('name')}\" looks like it needs some cleanup before you can trust it.")

    top_findings = [
        f"You have {summary.get('rows')} rows of data across {summary.get('columns')} columns — plenty to work with.",
        f"Data quality scores {summary.get('quality_score')} out of 100 — {'solid' if (summary.get('quality_score') or 0) >= 70 else 'could use attention'}.",
        "A few columns need cleanup before you can confidently report on them.",
    ]

    return {
        "whats_in_this_data": (
            f"{metadata.get('file_name', 'This file')} contains records you can use for trend and category analysis "
            "once a few quick cleanup items are addressed."
        ),
        "top_findings": top_findings,
        "watch_out_for": warnings[:2],
        "suggested_next_step": "Start by reviewing the columns flagged in red or yellow, then re-check your summaries.",
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
        "You are an expert, approachable data consultant — a friendly 'statistician at hand' "
        "translating raw numbers into plain English for a non-technical business user.\n\n"
        "STRICT RULES:\n"
        "- ZERO statistical jargon. Never say: p-value, correlation coefficient, standard deviation, "
        "IQR, skewness, variance, regression, z-score, percentile, kurtosis.\n"
        "- USE INSTEAD: average, typical range, strong relationship, unusual data points, trend.\n"
        "- Do NOT read numbers back. Translate the math into real-world behavior — focus on the 'so what?'\n"
        "- If column names are unclear (e.g. '150', '4'), refer to them by name but gently note that "
        "renaming them would give better context.\n"
        "- Every claim must be grounded in the computed facts provided. Never invent numbers.\n\n"
        "OUTPUT STRUCTURE (map to JSON keys):\n"
        "- whats_in_this_data: 'The Headline' — a punchy, one-sentence summary of the most important finding.\n"
        "- top_findings: 'What's Happening' — 2-3 bullet points explaining major trends in conversational English.\n"
        "- watch_out_for: risks or data quality issues phrased as friendly warnings (up to 2).\n"
        "- suggested_next_step: 'What You Should Do' — one highly actionable recommendation.\n"
    )
    user_prompt = (
        _computed_facts_block(analysis, metadata)
        + "\n\nReturn strict JSON with keys: whats_in_this_data (string), top_findings (array len 3), watch_out_for (array len up to 2), suggested_next_step (string)."
    )
    return system_prompt, user_prompt


def _column_prompt(column_name: str, stats: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are a friendly data consultant describing one column to a non-technical business user.\n\n"
        "STRICT RULES:\n"
        "- ZERO jargon. Never say: standard deviation, IQR, skewness, kurtosis, variance, z-score, "
        "percentile, regression, correlation coefficient.\n"
        "- USE INSTEAD: average, typical range, spread, unusual data points, consistent/inconsistent.\n"
        "- Humanize the data: tell a story about its shape, don't just list numbers.\n"
        "  Instead of 'Mean is 5.8, standard deviation is 0.83' say: "
        "  'Most values sit comfortably around 5.8. The data is quite consistent, mostly staying between 5.0 and 6.6.'\n"
        "- If the column name is unclear, refer to it by name but suggest renaming for clarity.\n"
        "- Use only supplied computed values. Never invent numbers.\n"
    )
    if stats.get("inferred_type") == "numeric":
        user_prompt = (
            f"Column: \"{column_name}\"\n"
            f"Computed facts: shape={stats.get('distribution_shape')}, average={stats.get('mean')}, midpoint={stats.get('median')}, "
            f"most-common={stats.get('mode')}, spread={stats.get('std')}, "
            f"range={stats.get('min')} to {stats.get('max')}, unusual-points={stats.get('outliers_iqr_count')}, missing={stats.get('missing_pct')}%\n\n"
            "Describe this column in 3 short, conversational paragraphs:\n"
            "(1) Where values cluster and how spread out they are — tell a story, don't recite numbers.\n"
            "(2) Whether anything stands out or seems off.\n"
            "(3) One practical thing the user should do about it.\n\n"
            "Return JSON keys: what_does_this_look_like, anything_unusual, what_should_i_do."
        )
    else:
        top = stats.get("top_10", [])[:5]
        user_prompt = (
            f"Column: \"{column_name}\"\n"
            f"Top categories: {top}\n"
            f"Total different values: {stats.get('cardinality')}\n"
            f"Missing: {stats.get('missing_pct')}%\n\n"
            "Describe this column in 2 short, conversational paragraphs:\n"
            "(1) What makes up this column — is it dominated by a few values or evenly spread?\n"
            "(2) What this column might be useful for, or any concerns.\n\n"
            "Return JSON keys: what_does_this_look_like, anything_unusual, what_should_i_do."
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
            "You are a friendly 'statistician at hand' — an approachable data consultant helping "
            "a non-technical business user understand their data.\n\n"
            "RULES:\n"
            "- ZERO jargon. Never say: p-value, correlation coefficient, standard deviation, IQR, "
            "skewness, variance, regression, z-score, percentile. "
            "Use plain words: average, typical range, strong relationship, unusual points, trend.\n"
            "- Focus on the 'so what?' — translate math into real-world behavior.\n"
            "- Answer using ONLY the computed facts provided. Never invent numbers.\n"
            "- If a requested fact is absent, say: 'I don't have that information from this dataset.'\n"
            "- Use actual column names and values. Keep answers to 2-4 conversational sentences."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response).strip()
    return text or "I couldn't generate a response right now."
