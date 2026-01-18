"""
Main FastAPI application entry point.
"""
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.routes import api_router
from app.utils.timing import Timer

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate request IDs and log request/response timing."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Attach to request state
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        # Measure request latency
        with Timer() as timer:
            # Process request
            response = await call_next(request)
        
        # Add request ID to response header
        response.headers["X-Request-Id"] = request_id
        
        # Log request end with latency
        latency_ms = timer.elapsed_ms
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"status={response.status_code} latency_ms={latency_ms:.2f}",
            extra={"request_id": request_id}
        )
        
        return response


# Create FastAPI app
app = FastAPI(
    title="MedGemma Clinical Note Assistant API",
    version=settings.VERSION,
    description=(
        "Offline clinical note generation API using MedGemma models. "
        "Supports generating SOAP notes, discharge summaries, and referral letters "
        "from clinical input data with strict formatting and PHI protection."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Add request ID middleware (before CORS to ensure it processes all requests)
app.add_middleware(RequestIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MedGemma Note Backend API",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
