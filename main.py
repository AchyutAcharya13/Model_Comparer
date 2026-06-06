from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
from trainer import (
    run_training,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_TEST_SUBSET_SIZE,
    DEFAULT_TRIALS,
)

app = FastAPI(title="CIFAR-10 Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_OPTIMIZERS = {"adam", "sgd", "adabs"}
OPTIMIZER_ALIASES = {"adabound": "adabs"}
FRONTEND_DIR = Path(__file__).parent / "frontend"


class TrainRequest(BaseModel):
    optimizer: str = "adam"
    epochs: int = DEFAULT_NUM_EPOCHS
    subset_size: int = DEFAULT_SUBSET_SIZE
    test_subset_size: int = DEFAULT_TEST_SUBSET_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED
    deterministic: bool = True
    trials: int = DEFAULT_TRIALS
    full_dataset: bool = False
    use_augmentation: bool = True
    paper_protocol: bool = False
    optimizer_params: dict[str, Any] = Field(default_factory=dict)


class EpochResult(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class SummaryResult(BaseModel):
    num_trials: int
    final_train_loss_mean: float
    final_train_loss_std: float
    final_train_acc_mean: float
    final_train_acc_std: float
    final_val_loss_mean: float
    final_val_loss_std: float
    final_val_acc_mean: float
    final_val_acc_std: float


class TrialResult(BaseModel):
    run_id: str
    trial: int
    optimizer: str
    seed: int
    deterministic: bool
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    history: list[EpochResult]
    checkpoint_path: str
    latest_checkpoint_path: str
    artifact_path: str
    timestamp: str


class TrainResponse(BaseModel):
    optimizer: str
    epochs: int
    subset_size: int
    test_subset_size: int
    batch_size: int
    full_dataset: bool
    use_augmentation: bool
    paper_protocol: bool
    trials: int
    seed: int
    deterministic: bool
    optimizer_params: dict[str, Any]
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    history: list[EpochResult]
    summary: SummaryResult
    trial_results: list[TrialResult]
    artifact_path: str
    trial_artifact_paths: list[str]
    checkpoint_path: str
    latest_checkpoint_path: str


class CompareRequest(BaseModel):
    epochs: int = DEFAULT_NUM_EPOCHS
    subset_size: int = DEFAULT_SUBSET_SIZE
    test_subset_size: int = DEFAULT_TEST_SUBSET_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED
    deterministic: bool = True
    trials: int = DEFAULT_TRIALS
    full_dataset: bool = False
    use_augmentation: bool = True
    paper_protocol: bool = False
    optimizer_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    epochs: int
    subset_size: int
    test_subset_size: int
    batch_size: int
    trials: int
    seed: int
    deterministic: bool
    full_dataset: bool
    use_augmentation: bool
    paper_protocol: bool
    best_optimizer: str
    results: list[TrainResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/optimizers")
def optimizers():
    return {"optimizers": sorted(VALID_OPTIMIZERS)}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    requested_name = req.optimizer.lower()
    name = OPTIMIZER_ALIASES.get(requested_name, requested_name)
    if name not in VALID_OPTIMIZERS:
        allowed = sorted(VALID_OPTIMIZERS | set(OPTIMIZER_ALIASES.keys()))
        raise HTTPException(status_code=400, detail=f"Optimizer must be one of: {allowed}")
    max_epochs = 500 if req.paper_protocol else 50
    if req.epochs < 1 or req.epochs > max_epochs:
        raise HTTPException(status_code=400, detail=f"epochs must be between 1 and {max_epochs}")
    if req.batch_size < 1:
        raise HTTPException(status_code=400, detail="batch_size must be >= 1")
    if req.trials < 1 or req.trials > 20:
        raise HTTPException(status_code=400, detail="trials must be between 1 and 20")
    if not req.full_dataset and (req.subset_size < 128 or req.test_subset_size < 128):
        raise HTTPException(status_code=400, detail="subset_size and test_subset_size must be >= 128")
    try:
        result = run_training(
            name,
            num_epochs=req.epochs,
            subset_size=req.subset_size,
            test_subset_size=req.test_subset_size,
            batch_size=req.batch_size,
            seed=req.seed,
            deterministic=req.deterministic,
            trials=req.trials,
            full_dataset=req.full_dataset,
            use_augmentation=req.use_augmentation,
            paper_protocol=req.paper_protocol,
            optimizer_params=req.optimizer_params,
            save_model=True,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    max_epochs = 500 if req.paper_protocol else 50
    if req.epochs < 1 or req.epochs > max_epochs:
        raise HTTPException(status_code=400, detail=f"epochs must be between 1 and {max_epochs}")
    if req.batch_size < 1:
        raise HTTPException(status_code=400, detail="batch_size must be >= 1")
    if req.trials < 1 or req.trials > 20:
        raise HTTPException(status_code=400, detail="trials must be between 1 and 20")
    if not req.full_dataset and (req.subset_size < 128 or req.test_subset_size < 128):
        raise HTTPException(status_code=400, detail="subset_size and test_subset_size must be >= 128")

    try:
        results = []
        for optimizer_name in sorted(VALID_OPTIMIZERS):
            outcome = run_training(
                optimizer_name,
                num_epochs=req.epochs,
                subset_size=req.subset_size,
                test_subset_size=req.test_subset_size,
                batch_size=req.batch_size,
                seed=req.seed,
                deterministic=req.deterministic,
                trials=req.trials,
                full_dataset=req.full_dataset,
                use_augmentation=req.use_augmentation,
                paper_protocol=req.paper_protocol,
                optimizer_params=req.optimizer_params.get(optimizer_name, {}),
                save_model=True,
            )
            results.append(outcome)

        best = max(results, key=lambda item: item["final_val_acc"])
        return {
            "epochs": req.epochs,
            "subset_size": req.subset_size,
            "test_subset_size": req.test_subset_size,
            "batch_size": req.batch_size,
            "trials": req.trials,
            "seed": req.seed,
            "deterministic": req.deterministic,
            "full_dataset": req.full_dataset,
            "use_augmentation": req.use_augmentation,
            "paper_protocol": req.paper_protocol,
            "best_optimizer": best["optimizer"],
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_DIR / "index.html")
