# SentinelQuant

SentinelQuant is an AI-assisted market intelligence platform for Indian equities. It combines NSE and market data ingestion, technical and fundamental features, sentiment signals, ensemble models, LSTM forecasting, market-regime detection, and GARCH-filtered Monte Carlo analysis.

The repository contains a React/Vite frontend and a FastAPI backend backed by PostgreSQL/TimescaleDB, Redis, and MLflow.

> **Important:** SentinelQuant is an analytical and advisory tool. It does not execute trades and its output is not SEBI-registered investment advice. Past performance does not guarantee future results.

## What it provides

- Daily stock recommendations with score, direction, confidence, reasoning, sector, and market-cap filters
- Technical, fundamental, and sentiment signal breakdowns
- Historical OHLCV data, quotes, stock screening, sector summaries, and market overview data
- Bull, bear, and volatile market-regime classification
- GARCH(1,1)-filtered Monte Carlo probability intervals
- Walk-forward backtesting and model-drift reporting
- JWT authentication with RS256 tokens, role-based admin routes, rate limiting, and security headers
- Scheduled data refresh, feature computation, inference, and retraining workflows

## ML and risk models

SentinelQuant uses several models with different jobs rather than relying on one model to make every decision.

### 1. XGBoost and LightGBM: primary directional model

The short-horizon model predicts the probability that a stock's 5-day log return will be positive. It trains a LightGBM classifier and an XGBoost classifier for each market-cap bucket:

- `large`: large-cap stocks
- `mid`: mid-cap stocks
- `small`: small-cap stocks

The two tree models are averaged to produce the bucket-level probability. Tree models are useful here because they handle mixed technical, fundamental, and sentiment features and can work with missing values. The current feature set includes RSI, MACD, Bollinger bandwidth, 200-day moving-average regime, ADX, ATR, free-cash-flow yield, P/E z-score, debt-to-equity ratio, 24-hour and 72-hour sentiment, and 3-month volume z-score.

Hyperparameters can be tuned with Optuna. Model artifacts are versioned and stored separately for each market-cap bucket so that large-, mid-, and small-cap stocks are not forced through the same model.

### 2. LSTM: secondary medium-horizon model

The PyTorch LSTM is designed for a 20-day directional target. It reads a rolling 30-day sequence of the same engineered features and outputs the probability of a positive 20-day log return. The network uses two LSTM layers, a hidden size of 128, dropout, L2 weight decay, early stopping, and train-window-only normalization.

The LSTM is treated as a secondary signal because sequence models can overfit limited financial datasets. It should be evaluated against a simpler time-series baseline and only used when validation shows that it adds value beyond the primary tree ensemble.

### 3. GARCH-filtered Monte Carlo: probabilistic risk model

The risk model fits a GARCH(1,1) process to recent returns, using Student-t innovations to account for volatility clustering and fat-tailed returns. It then runs 10,000 simulated paths across 5- and 20-day horizons. Each path applies a daily +/-20% circuit-breaker limit.

The output includes the probability of a positive return, median expected return, a 5% VaR proxy, 95% confidence bounds, simulated price statistics, and the number of paths used. This model is a risk and uncertainty filter; it does not replace the directional classifier.

### 4. Ensemble and regime controls

The recommendation layer combines the available signals using baseline weights of 50% XGBoost/LightGBM, 30% LSTM, and 20% GARCH-Monte Carlo. If a secondary signal is unavailable, its weight is redistributed to the primary model. The final score is clamped to 0-100 and produces only `long` or `neutral` directions.

Market regime detection uses the Nifty 50's 200-day moving average, India VIX when available, and ADX confirmation to classify conditions as bull, bear, or volatile. Weights are adjusted by regime and market-cap bucket, and high debt-to-equity ratios can force a neutral result. A reinforcement-learning position-sizing layer is documented as a future phase; it is not currently responsible for generating buy or sell signals.

### Validation principles

- Use walk-forward validation for time-dependent data; do not randomly shuffle observations across train and test sets.
- Fit normalizers on the training window only to prevent look-ahead bias.
- Evaluate beyond accuracy with Sharpe ratio, maximum drawdown, win rate, calibration, and stability across market regimes.
- Keep model versions and recommendation outputs auditable through MLflow and the stored model artifacts.

## Architecture

```text
React + Vite frontend (port 3000 or 5173)
                                |
                                v
FastAPI backend (port 8000) ---- Redis (port 6379)
                                |
                                +-------------- PostgreSQL/TimescaleDB (port 5432)
                                |
                                +-------------- MLflow tracking server (port 5000)
```

## Prerequisites

- macOS or Linux
- Python 3.11 or newer
- Node.js 18 or newer and npm
- Docker Desktop with Docker Compose
- OpenSSL, for generating local JWT keys

## Quick start

1. Install frontend dependencies:

      ```bash
      npm install
      ```

2. Create the backend virtual environment and install Python dependencies:

      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      pip install -r backend/requirements.txt
      ```

3. Generate the local RS256 key pair:

      ```bash
      ./scripts/generate_jwt_keys.sh
      ```

4. Create the backend environment file:

      ```bash
      cp backend/.env.example backend/.env
      ```

      The defaults in `backend/.env.example` match the local Docker Compose services. Add data-provider API keys when required by your ingestion configuration.

5. Start PostgreSQL, Redis, and MLflow:

      ```bash
      docker compose -f docker/docker-compose.yml up -d postgres redis mlflow
      ```

6. Start the backend and frontend together:

      ```bash
      ./start.sh
      ```

Open the frontend at [http://localhost:5173](http://localhost:5173). The backend health endpoint is available at [http://localhost:8000](http://localhost:8000). In debug mode, interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

The launcher writes service logs to `logs/backend.log` and `logs/frontend.log`.

## Startup commands

```bash
./start.sh                 # Start infrastructure checks, backend, and frontend
./start.sh --backend       # Start only the FastAPI backend
./start.sh --frontend      # Start only the Vite frontend
./start.sh --pipeline      # Run the daily pipeline once, then exit
./start.sh --stop          # Stop services started by the launcher
```

For manual development:

```bash
# Backend
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend, in another terminal
npm run dev
```

## Configuration

Backend settings are loaded from environment variables or `backend/.env`. Important settings include:

| Variable | Purpose | Local default |
| --- | --- | --- |
| `DATABASE_URL` | Async PostgreSQL connection | `postgresql+asyncpg://nseai:nseai_dev_password@localhost:5432/nseai` |
| `DATABASE_URL_SYNC` | Sync PostgreSQL connection for pipelines | `postgresql://nseai:nseai_dev_password@localhost:5432/nseai` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `JWT_PRIVATE_KEY_PATH` | RS256 signing key | `docker/keys/jwt_private.pem` |
| `JWT_PUBLIC_KEY_PATH` | RS256 verification key | `docker/keys/jwt_public.pem` |
| `MLFLOW_TRACKING_URI` | MLflow tracking destination | `sqlite:///mlflow.db` |
| `CORS_ORIGINS` | JSON array of allowed frontend origins | `['http://localhost:3000']` |
| `DEBUG` | Enables debug API docs | `false` |

Never commit `backend/.env`, private keys, or provider credentials.

## API overview

All routes below are prefixed as shown. Protected routes require an access token in the `Authorization: Bearer <token>` header.

### Authentication

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/refresh`
- `GET /api/v1/auth/me`
- `POST /api/v1/auth/logout`

### Recommendations and analysis

- `GET /api/v1/recommendations`
- `GET /api/v1/recommendations/{symbol}`
- `GET /api/v1/market/regime`
- `GET /api/v1/stocks/{symbol}/signals`
- `GET /api/v1/stocks/{symbol}/montecarlo`

### Market data

- `GET /api/v1/market/quote/{symbol}`
- `POST /api/v1/market/quotes/batch`
- `GET /api/v1/market/historical/{symbol}?days=90`
- `GET /api/v1/market/stocks`
- `GET /api/v1/market/screener`
- `GET /api/v1/market/sectors`
- `GET /api/v1/market/overview`

### Admin

Admin access is required for operational endpoints:

- `GET /api/v1/admin/health`
- `POST /api/v1/admin/run-pipeline`
- `POST /api/v1/admin/retrain`
- `GET /api/v1/admin/backtest/{model}`
- `GET /api/v1/admin/drift`
- `GET /api/v1/admin/jobs`
- `GET /api/v1/admin/data-freshness`

## Data and model workflows

The daily pipeline fetches recent OHLCV data, recomputes stale features, generates recommendations, and updates market-regime data:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m backend.scripts.daily_pipeline
```

To skip the market-data fetch and run only downstream processing:

```bash
PYTHONPATH=. python -m backend.scripts.daily_pipeline --skip-fetch
```

Other workflow scripts include:

```bash
PYTHONPATH=. python -m backend.scripts.run_backfill
PYTHONPATH=. python -m backend.scripts.run_features
PYTHONPATH=. python -m backend.scripts.run_training
```

Model artifacts are stored under `backend/model_artifacts/`. The pipeline expects trained XGBoost/LightGBM artifacts for the large, mid, and small market-cap buckets before it can generate recommendations.

## Testing and builds

Run the backend tests with:

```bash
source .venv/bin/activate
pytest backend/tests
```

Build the frontend with:

```bash
npm run build
```

## Project layout

```text
src/                    React frontend
backend/api/            FastAPI route modules
backend/features/       Technical, fundamental, and sentiment features
backend/ingestion/      Market and news data fetchers
backend/models/         ML and regime models
backend/scripts/        Backfill, feature, training, and daily pipeline jobs
backend/training/       Model training and walk-forward validation
backend/tests/          Python tests
docker/                 Compose file, Dockerfile, and local service keys
scripts/                Database initialization and key generation
```

## Stopping local services

```bash
./start.sh --stop
docker compose -f docker/docker-compose.yml down
```

Use `docker compose ... down -v` only when you intentionally want to delete local database, Redis, and MLflow volumes.

