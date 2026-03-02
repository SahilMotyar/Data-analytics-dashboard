from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from itsdangerous import URLSafeTimedSerializer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.core.config import settings


serializer = URLSafeTimedSerializer(settings.share_secret)


def create_pdf_report(upload_id: str, analysis: dict[str, Any]) -> str:
    output_dir = settings.outputs_dir / upload_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "report.pdf"

    pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    pdf.setTitle("DataLens Report")
    text = pdf.beginText(40, 750)
    text.setFont("Helvetica", 11)

    summary = analysis.get("summary", {})
    lines = [
        "DataLens Analytics Report",
        f"Rows: {summary.get('rows')}",
        f"Columns: {summary.get('columns')}",
        f"Memory (MB): {summary.get('memory_mb')}",
        f"Quality Score: {summary.get('quality_score')}",
        "",
        "Dataset Overview:",
        analysis.get("dataset_overview", "No overview available."),
        "",
        "Columns:",
    ]

    for item in analysis.get("columns", []):
        lines.append(f"- {item['name']} ({item['inferred_type']})")

    for line in lines:
        text.textLine(line[:120])

    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    return f"/static/{upload_id}/{pdf_path.name}"


def create_share_link(upload_id: str) -> dict[str, Any]:
    token = serializer.dumps({"upload_id": upload_id})
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    return {
        "token": token,
        "expires_at": expires_at.isoformat(),
        "share_path": f"/shared/{token}",
    }
