"""
ML info router — GET /api/ml/info

Returns metadata about the current production model, benchmark results,
feature importances, and the full model comparison table.
"""

from fastapi import APIRouter, HTTPException

from backend.ml.predict import get_model_info

router = APIRouter(prefix="/ml", tags=["ml"])


@router.get("/info")
def ml_info() -> dict:
    """
    Return current ML model metadata.

    When benchmark_results.json is present (after running ml/benchmark.py),
    returns full metrics, feature importance, and model comparison table.

    When benchmark has not been run yet, returns a placeholder response with
    the default model description and a helpful message.
    """
    try:
        return get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML info error: {str(exc)}")
