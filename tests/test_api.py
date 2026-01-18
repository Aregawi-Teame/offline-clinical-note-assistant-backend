"""
Tests for API endpoints.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def mock_model_runner():
    """Mock ModelRunner that doesn't load real model."""
    with patch('app.api.v1.endpoints.generate.ModelRunner') as mock_class:
        mock_instance = Mock()
        mock_instance.run.return_value = "Mocked generated output"
        mock_instance.get_model_id.return_value = "google/medgemma-4b-it"
        mock_instance.is_model_loaded.return_value = False
        mock_instance.device = "cpu"
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_prompt_builder():
    """Mock PromptBuilder."""
    with patch('app.api.v1.endpoints.generate.PromptBuilder') as mock_class:
        mock_instance = Mock()
        mock_instance.build.return_value = "Mocked prompt"
        mock_class.return_value = mock_instance
        yield mock_instance


def test_health_endpoint_returns_ok():
    """Test that /api/v1/health returns ok status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelLoaded" in data
    assert "model" in data
    assert "device" in data


def test_generate_endpoint_success(mock_model_runner, mock_prompt_builder):
    """Test /api/v1/generate returns 200 with valid request."""
    # Setup mocks
    mock_model_runner.run.return_value = "SUBJECTIVE: Patient presents...\nOBJECTIVE: Vital signs...\nASSESSMENT: Diagnosis...\nPLAN: Treatment plan..."
    
    # Mock ResponseGuard to return the mocked output
    def mock_validate(response, task, original_prompt=None):
        return response
    
    with patch('app.api.v1.endpoints.generate.ResponseGuard') as mock_guard_class:
        mock_guard_instance = Mock()
        mock_guard_instance.validate.side_effect = mock_validate
        mock_guard_class.return_value = mock_guard_instance
        
        # Make request
        request_data = {
            "task": "SOAP",
            "notes": "Patient presents with chest pain. 45-year-old male.",
            "options": {
                "maxTokens": 800,
                "temperature": 0.2,
                "topP": 0.9
            }
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "SOAP"
        assert "output" in data
        assert "model" in data
        assert "latencyMs" in data
        assert "requestId" in data
        assert data["model"] == "google/medgemma-4b-it"


def test_generate_rejects_empty_notes():
    """Test that validation rejects empty notes."""
    request_data = {
        "task": "SOAP",
        "notes": "",  # Empty notes
        "options": {}
    }
    
    response = client.post("/api/v1/generate", json=request_data)
    
    assert response.status_code == 422  # Validation error from Pydantic
    data = response.json()
    assert "detail" in data


def test_generate_rejects_whitespace_only_notes():
    """Test that validation rejects whitespace-only notes."""
    request_data = {
        "task": "SOAP",
        "notes": "   \n\t  ",  # Only whitespace
        "options": {}
    }
    
    response = client.post("/api/v1/generate", json=request_data)
    
    assert response.status_code == 422  # Validation error from Pydantic
    data = response.json()
    assert "detail" in data


def test_soap_output_includes_required_headers(mock_model_runner, mock_prompt_builder):
    """Test that SOAP output must include all required headers."""
    # Mock output with all required SOAP headers
    complete_soap_output = """SUBJECTIVE:
Chief Complaint: Patient presents with chest pain
History of Present Illness: 45-year-old male with history of hypertension

OBJECTIVE:
Vital Signs: Blood pressure 140/90
Physical Examination: General appearance normal

ASSESSMENT:
Primary Diagnosis: Chest pain, rule out cardiac

PLAN:
Treatment Plan: Further cardiac evaluation
Follow-up: Cardiology consultation

Clarifying questions (max 3):
None required."""
    
    mock_model_runner.run.return_value = complete_soap_output
    
    # Mock ResponseGuard to pass through validation
    def mock_validate(response, task, original_prompt=None):
        # ResponseGuard should validate and return the response
        # In real scenario, it would check for headers, but for this test
        # we'll use a guard that accepts valid SOAP output
        from app.services.response_guard import ResponseGuard, REQUIRED_SECTIONS
        guard = ResponseGuard()
        
        # Check if all required sections are present
        required = REQUIRED_SECTIONS.get(task, [])
        response_upper = response.upper()
        for section in required:
            if f"{section}:" not in response_upper:
                raise ValueError(f"Missing required section: {section}")
        
        return response
    
    with patch('app.api.v1.endpoints.generate.ResponseGuard') as mock_guard_class:
        mock_guard_instance = Mock()
        mock_guard_instance.validate.side_effect = mock_validate
        mock_guard_class.return_value = mock_guard_instance
        
        request_data = {
            "task": "SOAP",
            "notes": "Patient presents with chest pain. 45-year-old male with history of hypertension.",
            "options": {}
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        output = data["output"].upper()
        
        # Verify all required SOAP headers are present
        assert "SUBJECTIVE:" in output
        assert "OBJECTIVE:" in output
        assert "ASSESSMENT:" in output
        assert "PLAN:" in output


def test_soap_output_missing_headers_returns_error(mock_model_runner, mock_prompt_builder):
    """Test that SOAP output missing required headers triggers validation error."""
    # Mock output missing required headers
    incomplete_output = "Patient presents with chest pain. Treatment plan includes rest."
    
    mock_model_runner.run.return_value = incomplete_output
    
    # ResponseGuard should reject this
    def mock_validate(response, task, original_prompt=None):
        from app.services.response_guard import ResponseGuard, REQUIRED_SECTIONS
        guard = ResponseGuard()
        
        # Check if all required sections are present
        required = REQUIRED_SECTIONS.get(task, [])
        response_upper = response.upper()
        missing = []
        for section in required:
            if f"{section}:" not in response_upper:
                missing.append(section)
        
        if missing:
            raise ValueError(f"Missing required sections: {', '.join(missing)}")
        
        return response
    
    with patch('app.api.v1.endpoints.generate.ResponseGuard') as mock_guard_class:
        mock_guard_instance = Mock()
        mock_guard_instance.validate.side_effect = mock_validate
        mock_guard_class.return_value = mock_guard_instance
        
        request_data = {
            "task": "SOAP",
            "notes": "Patient presents with chest pain.",
            "options": {}
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        
        # Should return 400 validation error
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "Missing required sections" in data["error"]["message"]
