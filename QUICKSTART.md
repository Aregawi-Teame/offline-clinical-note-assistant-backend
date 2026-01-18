# Quick Start Guide

## Quick Setup (Recommended: Demo Mode)

### 1. Setup Environment

```bash
# Copy environment file
cp .env.example .env

# Edit .env and set DEMO_MODE=true for fast testing (no model download)
echo "DEMO_MODE=true" >> .env
```

### 2. Local Run (Virtual Environment)

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8085
```

### 3. Or Use Docker

```bash
# Build and run
docker-compose up --build
```

### 4. Or Use Google Colab Pro (with GPU)

For GPU acceleration on Google Colab, use the provided notebook:

1. **Open the Colab notebook**: `medgemma_colab_setup.ipynb`
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Follow the notebook steps**:
   - Install dependencies
   - Upload your `app/` directory (or clone from GitHub)
   - Configure environment (set `DEVICE=auto` for CUDA)
   - Start server with ngrok (gets public URL)
   - Test the API

**Benefits of Colab:**
- ✅ Free/cheap GPU access (~$10/month for Pro)
- ✅ Automatic CUDA detection (`DEVICE=auto` uses GPU)
- ✅ Public URL via ngrok
- ✅ No local setup needed

**Note:** Colab sessions timeout after 12-24 hours. For production, consider Docker or cloud GPU services.

## Testing the API

### 1. Health Check

```bash
curl http://localhost:8085/api/v1/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "modelLoaded": false,
  "model": "google/medgemma-4b-it",
  "device": "cpu"
}
```

### 2. Generate SOAP Note (Demo Mode)

```bash
curl -X POST "http://localhost:8085/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "SOAP",
    "notes": "Patient presents with chest pain. 45-year-old male with history of hypertension.",
    "options": {
      "maxTokens": 800,
      "temperature": 0.2,
      "topP": 0.9
    }
  }'
```

### 3. Generate Discharge Summary

```bash
curl -X POST "http://localhost:8000/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "DISCHARGE",
    "notes": "Patient admitted for pneumonia. Course was uncomplicated.",
    "options": {}
  }'
```

### 4. Generate Referral Letter

```bash
curl -X POST "http://localhost:8000/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "REFERRAL",
    "notes": "Patient needs cardiology consultation for chest pain evaluation.",
    "options": {
      "specialty": "cardiology"
    }
  }'
```

## Interactive API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## Running Tests

```bash
# Install dev dependencies if not already installed
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=app tests/
```

## Demo Mode vs Real Model

### Demo Mode (Fast, No Model Required)
```bash
# In .env
DEMO_MODE=true
```
- ✅ Instant responses
- ✅ No model download needed
- ✅ Great for UI development
- ✅ Fast testing

### Real Model Mode (Requires Model)
```bash
# In .env
DEMO_MODE=false
MODEL_ID=google/medgemma-4b-it
DEVICE=auto  # or cpu, cuda, mps
```
- ✅ Uses actual model inference
- ⚠️ Requires model download (~4GB for medgemma-4b-it)
- ⚠️ First request slower (model loading)
- ⚠️ Requires transformers and torch libraries

**Note:** To use real model, you'll need to install additional dependencies:
```bash
pip install transformers torch
```

## Troubleshooting

### Port Already in Use
```bash
# Change port in .env
PORT=8001

# Or kill process on port 8000
lsof -ti:8000 | xargs kill
```

### Module Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -e .
```

### Database Permission Error (if using audit logging)
```bash
# Ensure data directory is writable
mkdir -p data
chmod 755 data
```

## Example Complete Test Flow

```bash
# 1. Start server (in one terminal)
uvicorn app.main:app --reload

# 2. Check health (in another terminal)
curl http://localhost:8085/api/v1/health | jq

# 3. Generate a note
curl -X POST "http://localhost:8085/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{"task": "SOAP", "notes": "Test note"}' | jq

# 4. Check response includes required sections
# Look for SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN in output
```

## Google Colab Pro Setup (GPU Support)

For GPU acceleration without local hardware, use Google Colab Pro:

### Quick Start

1. **Open notebook**: Open `medgemma_colab_setup.ipynb` in Google Colab
2. **Enable GPU**: Runtime → Change runtime type → **GPU (T4 or better)**
3. **Run all cells** in order

### What You Get

- ✅ **GPU Access**: T4 or A100 GPU (with Colab Pro)
- ✅ **Automatic CUDA**: `DEVICE=auto` detects and uses CUDA
- ✅ **Public URL**: ngrok provides accessible endpoint
- ✅ **No Local Setup**: Everything runs in Colab cloud

### Configuration

The notebook sets these automatically:
```python
os.environ['DEVICE'] = 'auto'      # Auto-detects CUDA
os.environ['MODEL_ID'] = 'google/medgemma-4b-it'
os.environ['DEMO_MODE'] = 'false'  # Use real model
```

### Using the Public URL

After running the notebook, you'll get a public URL like:
```
https://xxxx-xxxx-xxxx.ngrok-free.app
```

Use this URL from anywhere:
```bash
# Health check
curl https://xxxx-xxxx-xxxx.ngrok-free.app/api/v1/health

# Generate note
curl -X POST "https://xxxx-xxxx-xxxx.ngrok-free.app/api/v1/generate/" \
  -H "Content-Type: application/json" \
  -d '{"task": "SOAP", "notes": "Patient presents with..."}'
```

### Important Notes

- **Session Limits**: Colab free (12h), Pro (24h max)
- **ngrok URL**: Changes on restart (free tier)
- **Model Loading**: First request takes ~30-60 seconds
- **Keep Running**: Keep the notebook open to maintain the server

### Troubleshooting Colab

**GPU Not Available?**
- Ensure you selected GPU runtime type
- Free Colab may not always provide GPU
- Colab Pro guarantees GPU access

**Module Not Found?**
- Check that you uploaded the `app/` directory
- Verify all files are in the correct structure

**Server Won't Start?**
- Check cell output for error messages
- Ensure all dependencies installed correctly
- Try setting `DEMO_MODE=true` to test without model
