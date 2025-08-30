"""Simplified main application to test backend startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Quantum Trading AI - Simple",
    description="Simplified backend for testing",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Quantum Trading AI Backend is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quantum-trading-ai"}

@app.get("/api/v1/test")
async def test_endpoint():
    """Test endpoint."""
    return {"message": "API is working!", "version": "v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

