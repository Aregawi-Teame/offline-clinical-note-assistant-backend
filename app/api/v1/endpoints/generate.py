"""
Generate endpoint for clinical note generation.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.schemas.generate import GenerateRequest, GenerateResponse, TaskEnum
from app.services.audit_logger import AuditLogger
from app.services.prompt_builder import PromptBuilder
from app.services.model_runner import ModelRunner
from app.services.response_guard import ResponseGuard
from app.utils.timing import Timer

router = APIRouter()
logger = get_logger(__name__)

# Initialize audit logger (will only be active if STORE_REQUESTS=true)
audit_logger = AuditLogger()


def get_response_guard(model_runner: ModelRunner = Depends(ModelRunner)) -> ResponseGuard:
    """
    Dependency function to create ResponseGuard with ModelRunner.
    
    Args:
        model_runner: ModelRunner instance
        
    Returns:
        ResponseGuard instance
    """
    return ResponseGuard(model_runner=model_runner)


def _create_error_response(
    code: str,
    message: str,
    request_id: str,
    status_code: int
) -> JSONResponse:
    """
    Create a consistent JSON error response.
    
    Args:
        code: Error code
        message: Error message
        request_id: Request ID
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error format
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "requestId": request_id
            }
        }
    )


@router.post("/", response_model=GenerateResponse)
async def generate_note(
    request_body: GenerateRequest,
    http_request: Request,
    prompt_builder: PromptBuilder = Depends(PromptBuilder),
    model_runner: ModelRunner = Depends(ModelRunner),
    response_guard: ResponseGuard = Depends(get_response_guard),
) -> GenerateResponse:
    """
    Generate a clinical note based on input data.
    
    Args:
        request_body: Generate request containing task, notes, and options
        http_request: FastAPI request object for accessing request_id
        prompt_builder: Prompt builder service
        model_runner: Model runner service
        response_guard: Response guard service
        
    Returns:
        GenerateResponse with generated note and metadata
        
    Raises:
        JSONResponse: Error responses with consistent format
    """
    # Get request ID from middleware
    request_id = getattr(http_request.state, "request_id", None) or "unknown"
    
    try:
        with Timer() as timer:
            # Build prompt
            try:
                specialty = request_body.options.specialty if request_body.options else None
                prompt = prompt_builder.build(
                    task=request_body.task,
                    notes=request_body.notes,
                    specialty=specialty
                )
            except ValueError as e:
                # Template or validation errors
                logger.warning(f"Prompt building failed: {e}", extra={"request_id": request_id})
                return _create_error_response(
                    code="VALIDATION_ERROR",
                    message=f"Failed to build prompt: {str(e)}",
                    request_id=request_id,
                    status_code=400
                )
            
            # Prepare model options
            options = {}
            if request_body.options:
                options = {
                    "maxTokens": request_body.options.maxTokens,
                    "temperature": request_body.options.temperature,
                    "topP": request_body.options.topP,
                }
            
            # Run model
            try:
                raw_response = model_runner.run(prompt, options)
            except RuntimeError as e:
                # Model loading or inference errors
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.error(f"Model OOM error: {e}", extra={"request_id": request_id})
                    return _create_error_response(
                        code="MODEL_OOM_ERROR",
                        message="Model out of memory. Try reducing maxTokens or using CPU device.",
                        request_id=request_id,
                        status_code=503
                    )
                else:
                    logger.error(f"Model inference error: {e}", extra={"request_id": request_id})
                    return _create_error_response(
                        code="MODEL_ERROR",
                        message=f"Model inference failed: {str(e)}",
                        request_id=request_id,
                        status_code=503
                    )
            except ImportError as e:
                logger.error(f"Model dependencies missing: {e}", extra={"request_id": request_id})
                return _create_error_response(
                    code="MODEL_NOT_AVAILABLE",
                    message="Model dependencies not installed. Please install transformers and torch.",
                    request_id=request_id,
                    status_code=503
                )
            
            # Guard and validate response
            try:
                validated_response = response_guard.validate(
                    response=raw_response,
                    task=request_body.task,
                    original_prompt=prompt
                )
            except ValueError as e:
                # Response validation errors
                logger.warning(f"Response validation failed: {e}", extra={"request_id": request_id})
                return _create_error_response(
                    code="VALIDATION_ERROR",
                    message=f"Response validation failed: {str(e)}",
                    request_id=request_id,
                    status_code=400
                )
            
            # Calculate latency
            latency_ms = timer.elapsed_ms
            
            # Log to audit database if enabled
            audit_logger.log(
                request_id=request_id,
                task=request_body.task,
                notes=request_body.notes,
                output=validated_response,
                latency_ms=latency_ms
            )
            
            # Return response
            return GenerateResponse(
                task=request_body.task,
                output=validated_response,
                model=model_runner.get_model_id(),
                latencyMs=latency_ms,
                requestId=request_id
            )
            
    except Exception as e:
        # Unexpected errors
        logger.exception(
            f"Unexpected error during note generation: {e}",
            extra={"request_id": request_id}
        )
        return _create_error_response(
            code="INTERNAL_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
            request_id=request_id,
            status_code=500
        )
