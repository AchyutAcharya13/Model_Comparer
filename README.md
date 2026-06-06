# CIFAR-10 Training Service

This repository implements a small production-oriented machine learning service for CIFAR-10 training.
It exposes a FastAPI REST API for running training jobs and optimizer comparisons, while using PyTorch for model definition, training, evaluation, checkpointing, and artifact management.

The code demonstrates a lightweight training workflow with custom optimizer support, structured experiment output, and a simple backend for experimentation. It is currently a research/prototype service and requires additional production hardening before deployment.

## Production Readiness Improvements

### 1. Packaging and dependency management
- Add a `pyproject.toml` / `setup.cfg` or `requirements.in` for reproducible dependency installation.
- Pin exact package versions and remove broad version ranges.
- Add `pip-tools`, Poetry, or similar dependency management to prevent unexpected upgrades.
- Include a `.python-version` or environment documentation for supported Python release(s).

### 2. Configuration and environment handling
- Move hard-coded global defaults into a dedicated configuration system.
- Use environment variables for runtime settings like dataset paths, checkpoint directories, and model save locations.
- Add support for a config file or CLI options for training parameters.

### 3. Logging, monitoring, and observability
- Replace `print()` training output with structured logging via Python `logging` or a logging framework.
- Add request/response logging and performance metrics for API endpoints.
- Capture and expose training metrics, model accuracy, loss, runtime, and resource usage.

### 4. Error handling and validation
- Improve API error handling with custom exception handlers and consistent response formats.
- Add validation for optimizer parameter values and more robust input validation layers.
- Avoid returning raw exception messages directly in API error responses.

### 5. Testing and quality assurance
- Add unit tests for model, optimizer, training loop, and API endpoints.
- Add integration tests for the full `/train` and `/compare` workflow.
- Add linting and formatting checks (`flake8`, `ruff`, `black`, `isort`).
- Add static type coverage checks or stricter type linting for Pydantic and project code.

### 6. Security and API hardening
- Restrict CORS origins instead of allowing `*` in production.
- Add rate limiting, request size limits, and API authentication/authorization if needed.
- Sanitize and validate all user-provided input strictly.
- Run dependency security scans for known vulnerabilities.

### 7. Data and dataset handling
- Add dataset caching and dataset integrity checks.
- Support configurable data directories and optional pre-downloaded dataset storage.
- Add a production data loading pipeline with retries and error handling.

### 8. Model persistence and artifact management
- Use a well-organized storage layout for checkpoints and experiment artifacts.
- Add model versioning and metadata tracking for saved checkpoints.
- Support saving checkpoints to external storage (S3, Azure Blob, etc.) if running in cloud.
- Ensure deterministic artifact paths and cleanup policy for temporary files.

### 9. Training loop and performance
- Add support for mixed precision training and distributed training if needed.
- Add configurable `num_workers`, batch size scaling, and device selection defaults.
- Add early stopping, learning rate scheduling, and experiment reproducibility controls.
- Add deterministic training warnings and fallback behavior for non-deterministic environments.

### 10. Deployment and operations
- Add deployment manifests or scripts for Docker, Kubernetes, or cloud hosting.
- Add a `Dockerfile` and/or `docker-compose.yml` for local and production deployment.
- Add health checks, readiness probes, and graceful shutdown handling.
- Add documentation for how to launch the API service and run training jobs.

### 11. Documentation and developer experience
- Add a project README with setup, usage, API contract, and training examples.
- Document expected input payloads for `/train` and `/compare` endpoints.
- Add a contributing guide or developer notes for extending optimizers and model definitions.
- Add comments where the code behavior is not immediately obvious.

## Immediate action items
- Add `README.md` and a `requirements.txt` governance policy.
- Add unit and API tests.
- Add structured logging and remove `print()` output from the training loop.
- Add environment-aware config and secure CORS settings.

## Notes specific to this codebase
- The current API allows optimizer aliases and unbounded numeric input; tighten validation.
- The model is small and suitable for experimentation, but production usage should add model versioning and dataset validation.
- The current optimizer implementation is custom and should be tested thoroughly before being used in production.

