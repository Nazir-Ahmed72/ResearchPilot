# Deployment

ResearchPilot ships as one container image that runs three roles: the **API** (uvicorn), the
**worker** (Celery: ingestion, research and evaluation) and a one-off **migration** job
(Alembic). PostgreSQL is managed (Neon); Redis and Qdrant run as containers locally and can be
swapped for managed services by changing two URLs.

```mermaid
flowchart LR
    user[Browser / client] -->|":8000"| api
    subgraph compose["docker compose (network: backend)"]
        migrate[migrate<br/>alembic upgrade head<br/>one-off] -.->|before start| api
        migrate -.-> worker
        api[api<br/>uvicorn, non-root] -->|enqueue| redis[(redis<br/>AOF volume)]
        redis --> worker[worker<br/>celery, concurrency 1]
        api --> qdrant[(qdrant<br/>storage volume)]
        worker --> qdrant
        api --- data[(app-data volume<br/>uploaded PDFs)]
        worker --- data
        api --- hf[(hf-cache volume<br/>models)]
        worker --- hf
        pg[(postgres<br/>profile local-db)]
    end
    api --> neon[(Neon PostgreSQL)]
    worker --> neon
    migrate --> neon
    worker --> groq[Groq API]
    worker -. optional .-> langsmith[LangSmith]
    api -. optional .-> pg
```

## Image (`Dockerfile`)

| Property | Choice | Why |
|---|---|---|
| Base | `python:3.11-slim`, multi-stage | Small runtime; build tools stay in the builder stage |
| PyTorch | CPU wheel from the PyTorch index | The default wheel bundles CUDA (several GB) |
| Versions | Constrained by `requirements/lock.txt` | Same versions the tests ran with |
| Layer caching | Dependencies installed before the code is copied | Code changes rebuild in seconds |
| Models | Not baked in; downloaded on first use into `/app/.cache/hf` (a volume) | Image ~1.3 GB smaller; models persist across rebuilds |
| User | `app` (uid 10001), no shell | Not running as root |
| Health check | `GET /api/v1/health/live` | Liveness only, so a slow external dependency never restarts the API |
| Shutdown | `CMD` in exec form; uvicorn graceful timeout 30 s | SIGTERM reaches the server directly |

## Local stack (`docker-compose.yml`)

```bash
cp .env.example .env            # fill in GROQ_API_KEY and DATABASE_URL(_UNPOOLED)
docker compose up -d --build    # first build downloads dependencies (~5-10 min)
docker compose ps               # all services healthy?
open http://localhost:8000/docs
docker compose logs -f worker
docker compose down             # stop (data volumes are kept); `down -v` deletes them
```

| Service | Image | Health check | Restart | Notes |
|---|---|---|---|---|
| `api` | `researchpilot:local` | HTTP liveness | unless-stopped | Port 8000; memory limit 3 GB |
| `worker` | same | `python -m app.workers.healthcheck` (ready file + Redis ping) | unless-stopped | Queues `ingestion,research`; 120 s warm shutdown; memory limit 4 GB |
| `migrate` | same | – | no | `alembic upgrade head`; api and worker wait for it to complete |
| `redis` | `redis:8.8.4-alpine` | `redis-cli ping` | unless-stopped | Append-only persistence |
| `qdrant` | `qdrant/qdrant:v1.20.1` | TCP check on 6333 | unless-stopped | Matches qdrant-client 1.20 |
| `postgres` | `postgres:18.7-alpine` | `pg_isready` | unless-stopped | Only with `--profile local-db` |

Only the API port is published; Redis, Qdrant and PostgreSQL are reachable only inside the
`backend` network.

### Configuration precedence
* Everything comes from `.env` (`env_file`, optional).
* Compose then overrides `REDIS_URL` and `QDRANT_LOCATION` to the local containers, so a `.env`
  written for Upstash / Qdrant Cloud still works locally. To use the cloud services from
  containers, set `COMPOSE_REDIS_URL`, `COMPOSE_QDRANT_LOCATION` and `COMPOSE_QDRANT_API_KEY`.
* Fully offline database:
  ```bash
  COMPOSE_DATABASE_URL=postgresql+psycopg://researchpilot:researchpilot@postgres:5432/researchpilot \
  docker compose --profile local-db up -d --build
  ```
* With a local Qdrant the collection starts empty: upload papers again (`POST /papers/upload`).

### Graceful shutdown and recovery
* API: in-flight requests get 30 s.
* Worker: Celery's warm shutdown finishes the current task within `stop_grace_period` (120 s).
  If it is killed anyway, `acks_late` returns the task to the queue; ingestion is idempotent and
  research runs resume from their last LangGraph checkpoint.

## CI (`.github/workflows/ci.yml`)

| Job | What it checks |
|---|---|
| Lint | `ruff check`, `ruff format --check` |
| Tests | Full test suite on Python 3.10, 3.11 and 3.12 with CPU PyTorch and lock-file versions; no secrets, network services or paid APIs |
| Docker | `docker compose config` (both profiles); image build (BuildKit, GitHub Actions cache); smoke test: container starts, runs as non-root, `/health/live` and `/docs` respond, Alembic sees its migrations |

Docker is never run on the development machine: GitHub's runners build and test the image on
every push.

## Deploying to a cloud platform (not performed)

Nothing has been deployed and no paid resources were created. The image is ready for any
container platform. A minimal setup:

| Component | Render | AWS |
|---|---|---|
| API | Web Service from the Dockerfile (port 8000, health check path `/api/v1/health/live`) | ECS Fargate service behind an ALB |
| Worker | Background Worker, same image, start command = the compose `worker` command | Second ECS service, same task image |
| Migrations | Pre-deploy command `alembic upgrade head` | One-off ECS task before deploying |
| PostgreSQL | Neon (keep) | Neon, or RDS PostgreSQL |
| Redis | Render Key Value, or Upstash | ElastiCache, or Upstash |
| Qdrant | Qdrant Cloud | Qdrant Cloud, or Qdrant on ECS with EFS |
| Uploaded PDFs | **Persistent disk shared by API and worker is required** | **S3** (implement the 3-method storage interface) |
| Secrets | Environment groups | Secrets Manager / SSM |

Points to address before a production deployment:
* **File storage:** the API and worker exchange uploaded PDFs via a shared local volume. On
  platforms where they run on different machines, replace `LocalFileStorage` with an
  object-storage implementation (`save` / `read` / `delete`).
* **Authentication:** the API has no authentication yet. Add it as a FastAPI dependency
  (`app/api/dependencies.py`) before exposing it publicly.
* **LLM quota:** Groq's free tier (8k tokens/minute, 200k tokens/day for gpt-oss-120b) supports
  only light use; production needs a paid tier.
* **CPU sizing:** the reranker needs ~2 vCPU per worker for ~8 s reranks; more workers need more
  memory (~2–3 GB each with models loaded).
* **CORS / TLS:** terminate TLS at the load balancer; add CORS settings if a browser front end
  calls the API directly.
