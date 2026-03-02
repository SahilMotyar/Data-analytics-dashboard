"""
Nano Banana – API routes
========================
Single stateless endpoint: upload → analyse → return JSON → forget.
"""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.services.engine import analyze

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Accept a CSV or Excel file and return the full analysis payload."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    allowed = (".csv", ".xlsx", ".xls")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload one of: {', '.join(allowed)}",
        )

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # 50 MB limit
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50 MB.")

    try:
        result = analyze(contents, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while analysing your data: {str(exc)}",
        )

    return result
