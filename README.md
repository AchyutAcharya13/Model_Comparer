# Model Comparer: Optimizer Benchmarking Framework for CIFAR-10

## Overview

Model Comparer is a machine learning experimentation framework designed to evaluate and compare the performance of different optimization algorithms on image classification tasks.

The project trains deep learning models on the CIFAR-10 dataset and provides a systematic comparison of optimizers such as SGD, Adam, AdaBound, and custom adaptive optimization techniques. It records training metrics, generates experiment artifacts, stores model checkpoints, and provides a frontend dashboard for result visualization.

The framework enables researchers and developers to:

* Compare optimizer convergence behavior
* Analyze training and validation performance
* Benchmark custom optimization algorithms
* Visualize experiment results
* Save reproducible experiment artifacts

---

## Features

### Optimizer Comparison

Evaluate and benchmark multiple optimizers under identical training conditions:

* SGD
* Adam
* AdaBound
* Custom Adaptive Optimizers

### Experiment Tracking

Automatically records:

* Training loss
* Validation loss
* Accuracy
* Hyperparameters
* Runtime metrics

### Checkpoint Management

Save and restore model checkpoints for:

* Experiment reproducibility
* Model recovery
* Performance analysis

### Interactive Frontend

Includes a lightweight frontend for visualizing:

* Experiment results
* Optimizer performance
* Comparative statistics

### Artifact Generation

Stores detailed experiment outputs in JSON format for later analysis.

---

## Project Structure

```text
Model_Comparer/
│
├── main.py                 # Entry point
├── trainer.py              # Training pipeline
├── model.py                # Model architecture
├── optimizers.py           # Optimizer implementations
├── requirements.txt        # Python dependencies
├── README.md
├── python_files_brief.txt
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── data/                   # CIFAR-10 dataset (generated locally)
├── checkpoints/            # Saved model weights
└── artifacts/              # Experiment outputs
```

---

## Dataset

This project uses the CIFAR-10 dataset.

Dataset classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The dataset is not included in the GitHub repository due to size limitations.

### Download Dataset

Option 1:

Allow the training script to automatically download CIFAR-10.

Option 2:

Manually download CIFAR-10 from:

https://www.cs.toronto.edu/~kriz/cifar.html

Extract the dataset into:

```text
data/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/AchyutAcharya13/Model_Comparer.git

cd Model_Comparer
```

### Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train Models

```bash
python main.py
```

The system will:

1. Load CIFAR-10
2. Initialize model architecture
3. Configure optimizers
4. Train models
5. Save checkpoints
6. Generate experiment artifacts
7. Produce comparison metrics

---

## Frontend Dashboard

Open:

```text
frontend/index.html
```

in a browser.

The dashboard displays:

* Optimizer performance
* Training statistics
* Experiment summaries
* Comparative results

---

## Experiment Outputs

### Artifacts

Stored in:

```text
artifacts/
```

Contains:

* Experiment summaries
* Metrics
* Training results
* Optimizer comparisons

### Checkpoints

Stored in:

```text
checkpoints/
```

Contains:

* Model weights
* Training snapshots
* Best-performing models

---

## Supported Optimizers

| Optimizer         | Description                            |
| ----------------- | -------------------------------------- |
| SGD               | Stochastic Gradient Descent            |
| Adam              | Adaptive Moment Estimation             |
| AdaBound          | Adaptive optimizer with dynamic bounds |
| Custom Optimizers | Experimental optimization algorithms   |

---

## Use Cases

* Deep Learning Research
* Optimizer Benchmarking
* Academic Projects
* ML Experimentation
* Hyperparameter Analysis
* Custom Optimizer Development

---

## Future Improvements

## Future Improvements

* Support for additional datasets (CIFAR-100, ImageNet, Tiny ImageNet)
* Benchmarking of modern architectures (ResNet, EfficientNet, Vision Transformers)
* Automated hyperparameter optimization using Optuna
* Multi-GPU and distributed training support
* Experiment tracking with MLflow and Weights & Biases
* Docker and Kubernetes deployment
* Real-time monitoring and analytics dashboards
* Explainable AI integration (Grad-CAM, SHAP)
* REST API deployment using FastAPI
* Cloud integration (AWS, Azure, Google Cloud)
* Model registry and version management
* Automated CI/CD pipelines for training and deployment

---

## Author

Achyut Acharya

GitHub:
https://github.com/AchyutAcharya13

---

## License

This project is intended for educational, research, and experimentation purposes.
