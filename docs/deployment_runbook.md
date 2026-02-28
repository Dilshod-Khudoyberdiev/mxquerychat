# Deployment Runbook (MVP)

This runbook covers a minimal deployment baseline for mxQueryChat.

## 1) Prerequisites
- Python 3.11+ available on target host.
- Local/self-hosted Ollama running and reachable.
- Required model pulled locally (default `mistral`).
- DuckDB file present: `mxquerychat.duckdb`.
- Repository checked out with `requirements.txt`.

## 2) Environment Variables
Set these values in your environment (or Azure App Settings):

- `OLLAMA_MODEL=mistral`
- `OLLAMA_URL=http://127.0.0.1:11434`
- `EXPLANATION_TIMEOUT_SECONDS=8`
- `OLLAMA_TIMEOUT_SECONDS=60`
- `APP_LOG_LEVEL=INFO`
- `APP_LOG_PATH=logs/app.log`
- `APP_METRICS_LOG_PATH=logs/metrics.jsonl`

Optional tuning:
- `OLLAMA_KEEP_ALIVE=30m`
- `OLLAMA_NUM_CTX=2048`
- `OLLAMA_NUM_PREDICT=96`
- `OLLAMA_TEMPERATURE=0`

## 3) Local Startup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## 4) Azure App Service Startup
Use this startup command:
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Ensure App Settings include the environment variables listed above.

## 5) Smoke Checks
1. App boot:
   - Open app URL and confirm `New Question` and `Training Data` views load.
2. SQL generation:
   - Ask a known question and confirm SQL appears.
3. Read-only safety:
   - Enter a write request (e.g., "delete rows") and confirm block.
4. Explanation behavior:
   - In Step 2, click `Generate Explanation`.
   - Confirm explanation appears, or timeout/error message appears without blocking query execution.

## 6) Logs and Metrics
- Application log: `logs/app.log`
- Metrics stream: `logs/metrics.jsonl`

Expected explanation telemetry event:
- `event = "sql_explanation"`
- fields include `success`, `path`, `failure_category`, `duration_ms`

## 7) Rollback
If deployment is unhealthy:
1. Revert to previous known-good commit.
2. Restart app process/service.
3. Re-run smoke checks above.
