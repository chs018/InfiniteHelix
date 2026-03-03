"""
Main FastAPI application for Infinite Helix.
AI-Powered Medical Report Analysis Platform.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import init_db, close_db
from app.routers import upload, analyze, results, auth, translate, chat
from app.schemas import ErrorResponse, HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Infinite Helix application...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
    
    logger.info(f"Application started in {settings.environment} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Infinite Helix application...")
    
    try:
        close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-Powered Medical Report Analysis - Democratizing Healthcare Intelligence",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with detailed error messages.
    """
    logger.warning(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later." if not settings.debug else str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).dict()
    )


# Include routers
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(results.router)
app.include_router(translate.router)
app.include_router(chat.router)


# Root endpoint
@app.get(
    "/",
    tags=["health"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def root(request: Request):
    """
    Root endpoint providing API information.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-Powered Medical Report Analysis Platform",
        "documentation": "/docs" if settings.debug else "Contact administrator",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Health Check",
    description="Check API health and service status"
)
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    # Check database connectivity
    db_healthy = True
    try:
        from app.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_healthy = False
    
    # Check services
    services_status = {
        "ocr": True,  # Could add actual OCR check
        "nlp": True,  # Could add actual NLP model check
    }
    
    return HealthCheckResponse(
        status="healthy" if db_healthy else "unhealthy",
        version=settings.app_version,
        database=db_healthy,
        services=services_status
    )


# Metrics endpoint (for monitoring)
@app.get(
    "/metrics",
    tags=["monitoring"],
    summary="Application Metrics",
    description="Get application metrics for monitoring"
)
async def metrics():
    """
    Return application metrics.
    """
    from app.database import SessionLocal
    from app.models import UploadedFile, Analysis
    
    try:
        db = SessionLocal()
        
        total_uploads = db.query(UploadedFile).count()
        total_analyses = db.query(Analysis).count()
        completed_analyses = db.query(Analysis).filter(
            Analysis.status == "completed"
        ).count()
        failed_analyses = db.query(Analysis).filter(
            Analysis.status == "failed"
        ).count()
        
        db.close()
        
        return {
            "total_uploads": total_uploads,
            "total_analyses": total_analyses,
            "completed_analyses": completed_analyses,
            "failed_analyses": failed_analyses,
            "success_rate": f"{(completed_analyses / total_analyses * 100):.2f}%" if total_analyses > 0 else "0%",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        return {
            "error": "Unable to retrieve metrics",
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
