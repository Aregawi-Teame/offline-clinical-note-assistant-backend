"""
Health check endpoint.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.model_runner import ModelRunner

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    modelLoaded: bool
    model: str
    device: str


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status, model loading state, model ID, and device
    """
    # Create ModelRunner to access device info (lightweight, no model loading)
    model_runner = ModelRunner()
    
    # Check if model is loaded (no heavy calls)
    model_loaded = model_runner.is_model_loaded()
    
    return HealthResponse(
        status="ok",
        modelLoaded=model_loaded,
        model=model_runner.get_model_id(),
        device=model_runner.device
    )
