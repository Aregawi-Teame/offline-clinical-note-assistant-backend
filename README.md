# MedGemma Note Backend

Backend API for generating clinical notes using MedGemma model.

## Features

- FastAPI-based REST API
- Support for multiple note types: SOAP, Discharge Summary, Referral Letters
- Template-based prompt generation
- Model response validation and guarding
- Comprehensive logging
- Health check endpoints

## Requirements

- Python 3.11+
- See `pyproject.toml` for dependencies

## Setup and Running

### Option 1: Local Development (Virtual Environment)

#### 1. Clone and Navigate

```bash
cd offline-clinical-note-assistant-backend
```

#### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

#### 4. Configure Environment

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings (model path, device, etc.).

#### 5. Run the Application

Using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python -m app.main
```

### Option 2: Docker

#### 1. Build and Run with Docker Compose

```bash
# Copy environment file (optional, will use defaults otherwise)
cp .env.example .env

# Build and start the container
docker-compose up --build
```

#### 2. Run with Docker (without compose)

```bash
# Build the image
docker build -t medgemma-backend .

# Run the container
docker run -p 8000:8000 --env-file .env medgemma-backend
```

**Note:** Model downloads happen at runtime (not during build) for faster builds and to leverage HuggingFace cache.

### Access the API

- API Documentation (Swagger): http://localhost:8000/api/v1/docs
- Alternative Docs (ReDoc): http://localhost:8000/api/v1/redoc
- Health Check: http://localhost:8000/api/v1/health
- Root: http://localhost:8000/

## Project Structure

```
medgemma-note-backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── v1/
│   │       ├── routes.py       # API router configuration
│   │       └── endpoints/
│   │           └── generate.py # Generate endpoint
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   └── logging.py          # Logging setup
│   ├── schemas/
│   │   └── generate.py         # Pydantic schemas
│   ├── services/
│   │   ├── prompt_builder.py   # Prompt template builder
│   │   ├── model_runner.py     # Model inference service
│   │   └── response_guard.py   # Response validation
│   ├── templates/
│   │   ├── soap.txt            # SOAP note template
│   │   ├── discharge.txt       # Discharge summary template
│   │   └── referral.txt        # Referral letter template
│   └── utils/
│       └── timing.py           # Timing utilities
├── tests/                       # Test directory
├── .env.example                # Example environment variables
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## API Usage

### Generate a Clinical Note

```bash
curl -X POST "http://localhost:8000/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "SOAP",
    "notes": "Patient presents with chest pain. 45-year-old male with history of hypertension. Blood pressure 140/90, heart rate regular.",
    "options": {
      "maxTokens": 800,
      "temperature": 0.2,
      "topP": 0.9
    }
  }'
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black app/
```

### Linting

```bash
ruff check app/
```

## Notes

- The `model_runner.py` service contains placeholder code for model loading and inference. You'll need to implement the actual model integration based on your model framework (e.g., transformers, llama.cpp, etc.).
- Update the `response_guard.py` validation logic based on your specific requirements.
- Adjust template files in `app/templates/` to match your prompt engineering needs.

## License

MIT
