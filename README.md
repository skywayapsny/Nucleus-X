# Nucleus-X

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

White-label local inference core for **GGUF** models with:

- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- A minimal custom generation endpoint: `/custom/v1/completions`
- Built-in guard rails for shell and path validation
- Optional API key protection
- In-memory rate limiting
- Health, readiness and metrics endpoints

The repository is intentionally small and reusable. Product naming and public model identity are controlled by environment variables, so the same core can be embedded into another app without renaming internals.

## Why this repo is useful

- Runs local GGUF inference behind a clean HTTP layer
- Preserves an inference abstraction through `Cassette`, so the backend is replaceable
- Exposes security policies as API-level validation endpoints instead of leaving them as dead utility code
- Adds operational basics expected from a publishable service: readiness, request IDs, latency headers and metrics

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`llama-cpp-python` is listed in `requirements.txt`, but the runtime is imported lazily. The server can start and pass smoke tests without loading the model until an inference request arrives.

## Configuration

Copy `.env.example` to `.env` and set at least:

```env
CORE_MODEL_PATH=/absolute/path/to/model.gguf
```

Useful optional variables:

- `API_TITLE`
- `SERVICE_NAME`
- `CORE_PUBLIC_MODEL_ID`
- `CORE_API_KEY`
- `CORE_RATE_LIMIT_RPM`
- `CORE_RATE_LIMIT_BURST`
- `CORE_ALLOWED_HOSTS`
- `CORE_EXPECTED_MODEL_SHA256`
- Sampling/runtime values from `.env.example`

## Run

```bash
python -m uvicorn adapter.main:app --host 127.0.0.1 --port 8000
```

Quick checks:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/models`

## Production baseline

- Set `CORE_API_KEY` to protect generation and security endpoints
- Set `CORE_ALLOWED_HOSTS` for host header protection
- Keep `CORE_EXPECTED_MODEL_SHA256` enabled when distributing model files across machines
- Run behind a reverse proxy with TLS termination

Container image:

```bash
docker build -t nucleus-x:latest .
docker run --rm -p 8000:8000 --env-file .env nucleus-x:latest
```

## OpenAI-compatible chat example

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer your-api-key" ^
  -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

If `CORE_API_KEY` is not set, the protected endpoints stay open for local development.

## Custom generation example

```bash
curl -s http://127.0.0.1:8000/custom/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Summarize local inference in one sentence\"}"
```

## Security validation examples

Validate a shell command without executing it:

```bash
curl -s http://127.0.0.1:8000/custom/v1/security/shell/validate ^
  -H "Content-Type: application/json" ^
  -d "{\"command\":\"git log --oneline -n 5\"}"
```

Validate a project path:

```bash
curl -s http://127.0.0.1:8000/custom/v1/security/path/validate ^
  -H "Content-Type: application/json" ^
  -d "{\"path\":\"README.md\",\"mode\":\"read\",\"must_exist\":true}"
```

## API surface

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /custom/v1/completions`
- `POST /custom/v1/security/shell/validate`
- `POST /custom/v1/security/path/validate`

## Architecture

- `adapter/main.py` exposes HTTP routes, operational middleware and validation endpoints
- `core/cassette.py` defines the backend contract
- `core/inference.py` provides the default GGUF engine
- `security/path_guard.py` validates project paths and write zones
- `security/shell_guards.py` classifies shell commands by risk

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall adapter core security tests
```

The test suite uses a mocked inference backend, so it does not require model weights.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- **Free for open-source use**: You may use, modify, and distribute this software freely under the terms of AGPL-3.0.
- **Copyleft requirement**: If you modify and deploy this software as a network service, you must provide the source code to your users.

### Commercial Licensing

For companies that wish to use Nucleus-X in proprietary or closed-source products without the AGPL-3.0 obligations, **commercial licenses are available**.

**Benefits of a commercial license:**
- Use in closed-source SaaS products
- No requirement to open-source your modifications
- Priority support and integration assistance
- Custom feature development

**Contact for pricing:**
- Email: [your-email@example.com]
- Or open a GitHub issue with the title "Commercial License Inquiry"

| License Type | Use Case | Source Code Disclosure |
|--------------|----------|------------------------|
| AGPL-3.0 (Free) | Open-source projects, personal use | Required |
| Commercial (Paid) | Proprietary SaaS, enterprise products | Not required |

## Legal Disclaimer

You are responsible for model weights, prompts, datasets and deployment compliance. This repository provides infrastructure only and does not ship third-party model files.
