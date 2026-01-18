"""
Pydantic schemas for generate endpoint.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TaskEnum(str, Enum):
    """Task types for clinical note generation."""
    
    SOAP = "SOAP"
    DISCHARGE = "DISCHARGE"
    REFERRAL = "REFERRAL"


class GenerateOptions(BaseModel):
    """Options for note generation."""
    
    specialty: Optional[str] = Field(
        default=None,
        description="Medical specialty (e.g., cardiology, neurology)"
    )
    maxTokens: int = Field(
        default=800,
        description="Maximum tokens to generate",
        ge=64,
        le=2048
    )
    temperature: float = Field(
        default=0.2,
        description="Sampling temperature",
        ge=0.0,
        le=1.0
    )
    topP: float = Field(
        default=0.9,
        description="Nucleus sampling top-p parameter",
        ge=0.0,
        le=1.0
    )


class GenerateRequest(BaseModel):
    """Request schema for note generation."""
    
    task: TaskEnum = Field(
        ...,
        description="Type of clinical note task to generate"
    )
    notes: str = Field(
        ...,
        description="Input clinical notes or data for generation",
        min_length=1,
        max_length=8000
    )
    options: Optional[GenerateOptions] = Field(
        default_factory=GenerateOptions,
        description="Generation options"
    )
    
    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: str) -> str:
        """Validate notes are non-empty (after stripping whitespace)."""
        if not v or not v.strip():
            raise ValueError("notes cannot be empty or whitespace only")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "task": "SOAP",
                "notes": "Patient presents with chest pain. 45-year-old male with history of hypertension. Blood pressure 140/90, heart rate regular. Assessment: Chest pain, rule out cardiac.",
                "options": {
                    "specialty": "cardiology",
                    "maxTokens": 800,
                    "temperature": 0.2,
                    "topP": 0.9
                }
            }
        }


class GenerateResponse(BaseModel):
    """Response schema for note generation."""
    
    task: TaskEnum = Field(..., description="Type of task that was generated")
    output: str = Field(..., description="Generated clinical note output")
    model: str = Field(..., description="Model identifier used for generation")
    latencyMs: float = Field(..., description="Generation latency in milliseconds")
    requestId: Optional[str] = Field(
        default=None,
        description="Request identifier for tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task": "SOAP",
                "output": "SUBJECTIVE: Patient presents with chest pain...\nOBJECTIVE: Blood pressure 140/90...\nASSESSMENT: Chest pain, rule out cardiac...\nPLAN: Further cardiac evaluation...",
                "model": "google/medgemma-4b-it",
                "latencyMs": 1234.56,
                "requestId": "req-12345"
            }
        }
