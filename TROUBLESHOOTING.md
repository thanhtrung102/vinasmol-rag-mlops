# Troubleshooting Guide

Common issues and their solutions for the VinaSmol RAG MLOps project.

## Training Issues

### FileNotFoundError: flash_attn_triton.py

**Error:**
```
FileNotFoundError: No such file or directory: '~/.cache/huggingface/modules/transformers_modules/vinai/PhoGPT_hyphen_4B_hyphen_Chat/.../flash_attn_triton.py'
```

**Cause:**
The PhoGPT model uses custom flash attention code that may have compatibility issues or cache corruption.

**Solutions:**

1. **Clear HuggingFace cache** (Recommended):
   ```bash
   make clean-hf-cache
   ```

   Or manually:
   ```bash
   rm -rf ~/.cache/huggingface/modules/transformers_modules/vinai/PhoGPT*
   ```

2. **Retry training**:
   ```bash
   make train-lora
   ```

The training script now automatically uses eager attention implementation instead of flash attention to avoid compatibility issues.

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in `configs/training_config.yaml`:
   ```yaml
   training:
     per_device_train_batch_size: 1  # Already minimum
     gradient_accumulation_steps: 16  # Increase this instead
   ```

2. **Enable gradient checkpointing** (already enabled by default):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Use CPU offload** (slower but uses less GPU memory):
   Add to model loading in `train_lora.py`:
   ```python
   device_map="balanced_low_0"  # Instead of "auto"
   ```

### ImportError: No module named 'mlflow'

**Error:**
```
ModuleNotFoundError: No module named 'mlflow'
```

**Cause:**
Dependencies not installed or using wrong Python environment.

**Solutions:**

1. **Run setup** (in Codespaces):
   ```bash
   make setup
   ```

2. **Install dependencies manually**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python environment**:
   ```bash
   which python
   python --version
   ```

## RAG Issues

### Qdrant Connection Error

**Error:**
```
requests.exceptions.ConnectionError: Failed to establish a connection
```

**Solutions:**

1. **Start services**:
   ```bash
   make services-up
   ```

2. **Check Qdrant health**:
   ```bash
   curl http://localhost:6333/health
   ```

3. **Restart Qdrant**:
   ```bash
   docker-compose restart qdrant
   ```

### Qdrant API Version Mismatch

**Error:**
```
404 Not Found on /collections/{name}/points/query
```

**Cause:**
Qdrant server version too old.

**Solution:**
Update `docker-compose.yaml`:
```yaml
qdrant:
  image: qdrant/qdrant:v1.11.3  # Or newer
```

### Collection Not Found

**Error:**
```
Collection 'vietnamese_docs' not found
```

**Solutions:**

1. **Create collection**:
   ```bash
   curl -X PUT http://localhost:6333/collections/vietnamese_docs \
     -H 'Content-Type: application/json' \
     -d '{
       "vectors": {
         "size": 384,
         "distance": "Cosine"
       }
     }'
   ```

2. **Run data pipeline**:
   ```bash
   make data-process
   make data-embed
   ```

## Evaluation Issues

### Ragas Requires OpenAI API Key

**Error:**
```
openai.error.AuthenticationError: No API key provided
```

**Solutions:**

1. **Set OpenAI API key** in `.env`:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Use local models** (experimental):
   Update `configs/eval_config.yaml`:
   ```yaml
   llm:
     model: "local_model_path"
   ```

### hallucination_detector ImportError

**Error:**
```
ImportError: cannot import name 'SimpleHallucinationDetector'
```

**Solutions:**

1. **Check imports** in `src/evaluation/__init__.py`
2. **Reinstall package**:
   ```bash
   pip install -e .
   ```

## API Issues

### FastAPI Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Kill existing process**:
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Use different port**:
   ```bash
   uvicorn src.api.main:app --port 8001
   ```

### Streaming Response Not Working

**Issue:**
Client not receiving streaming chunks.

**Solutions:**

1. **Use proper HTTP client**:
   ```bash
   curl -N http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "test"}'
   ```

2. **Check nginx/proxy settings** - disable buffering:
   ```nginx
   proxy_buffering off;
   ```

## Docker Issues

### docker-compose command not found

**Solutions:**

1. **Install Docker Compose**:
   ```bash
   sudo apt-get update
   sudo apt-get install docker-compose-plugin
   ```

2. **Use docker compose** (v2 syntax):
   ```bash
   docker compose up -d
   ```

### Permission Denied (Docker)

**Error:**
```
permission denied while trying to connect to Docker daemon
```

**Solutions:**

1. **Add user to docker group**:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Run with sudo**:
   ```bash
   sudo make services-up
   ```

## Environment Issues

### .env File Not Loading

**Solutions:**

1. **Check file location**:
   ```bash
   ls -la .env
   ```

2. **Copy from example**:
   ```bash
   cp .env.example .env
   ```

3. **Verify format** - no spaces around `=`:
   ```bash
   # Correct
   API_KEY=value

   # Incorrect
   API_KEY = value
   ```

## Getting Help

If you encounter an issue not listed here:

1. **Check logs**:
   ```bash
   docker-compose logs qdrant
   docker-compose logs mlflow
   ```

2. **Run tests**:
   ```bash
   make test-all
   ```

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Create an issue** on GitHub with:
   - Error message and full traceback
   - Steps to reproduce
   - Environment info (OS, Python version, GPU)
   - Output of `pip freeze`
