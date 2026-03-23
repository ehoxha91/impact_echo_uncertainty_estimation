# Impact Echo Uncertainty Estimation

Evidential Deep Learning for Impact Echo Signal Classification with Uncertainty Quantification

## Quick Start

### Setup Environment (Recommended: UV)

Create virtual environment:
```bash
chmod +x setup_env.sh && ./setup_env.sh
```

Or manually:
```bash
# Create a new virtual environment with Python 3.12.11
uv venv --python 3.12.11

# Activate the environment
source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv pip install -r requirements-uv.txt
```

### Train Model

```bash
python train_evidential_model.py
```

### Test Model
```bash
python test.py
```

## Project Structure

```
├── dataloaders/                  # Data loading and augmentation
├── models/
│   ├── evidential_model.py       # Evidential IENet architecture
│   ├── standard_model.py         # Standard IENet (softmax baseline)
│   ├── mc_dropout.py             # MC Dropout inference
│   └── deep_ensemble.py          # Deep Ensemble inference
├── losses/                       # Evidential loss functions
├── training/                     # Training and evaluation loops
├── testing/                      # Test analysis and visualization
├── train_evidential_model.py     # Train evidential model
├── train_baselines.py            # Train MC Dropout + Ensemble baselines
├── evaluate_uq_comparison.py     # UQ method comparison
├── evaluate_calibration.py       # ECE, Brier score, reliability diagrams
├── evaluate_multi_seed.py        # Multi-seed stability evaluation
├── evaluate_ablation.py          # Loss function ablation study
├── figures/                      # Generated figures (reliability diagrams)
├── data/                         # Training/test datasets
├── weights/                      # Saved model checkpoints (gitignored)
└── results/                      # Evaluation results
```

## Code Structure

1. **models/evidential_model.py** - Evidential IENet architecture
2. **models/standard_model.py** - Standard IENet (softmax baseline for MC Dropout / Ensembles)
3. **models/mc_dropout.py** - MC Dropout inference (T=50 forward passes)
4. **models/deep_ensemble.py** - Deep Ensemble inference (M=5 models)
5. **losses/evidential_loss.py** - Loss function implementations
6. **training/trainer.py** - Training and evaluation functions
7. **train_evidential_model.py** - Main evidential model training script
8. **train_baselines.py** - Train MC Dropout and Deep Ensemble baselines

## Using Components in Your Own Code

### Example 1: Create and use the model
```python
from models.evidential_model import create_model
import torch

# Create model
model = create_model(input_length=860, num_classes=2)

# Use it
x = torch.randn(32, 1, 860)  # batch_size=32, channels=1, length=860
evidence, defect_features = model(x)
```

### Example 2: Calculate loss
```python
from losses.evidential_loss import evidential_loss
import torch

evidence = torch.randn(32, 2)  # batch_size=32, num_classes=2
targets = torch.randint(0, 2, (32,))  # batch_size=32
epoch = 10

loss, nll, kl_div, penalty = evidential_loss(evidence, targets, epoch)
```

### Example 3: Train for one epoch
```python
from training.trainer import train_evidential_classifier

avg_loss, avg_nll, avg_kl, avg_penalty, accuracy = train_evidential_classifier(
    model, train_loader, optimizer, device, epoch, class_weights=None
)
```

### Example 4: Evaluate model
```python
from training.trainer import evaluate_evidential_classifier

results = evaluate_evidential_classifier(model, test_loader, device)
accuracy, predictions, uncertainties, epistemic, aleatoric, confidences, targets, alphas = results
```

## Environment Setup with UV

This project uses `uv` for fast Python package management. UV is a modern, Rust-based package manager that's significantly faster than pip/conda.

### Prerequisites

#### Install UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or with Homebrew:**
```bash
brew install uv
```

**Or with pip:**
```bash
pip install uv
```

For other installation methods, see: https://docs.astral.sh/uv/getting-started/installation/

### Installation Steps

#### 1. Create Virtual Environment

```bash
# Create a new virtual environment with Python 3.12.11
uv venv --python 3.12.11

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

#### 2. Install Dependencies

```bash
# Install PyTorch first (for MPS/CUDA support)
uv pip install torch torchvision torchaudio

# Install core dependencies
uv pip install numpy scipy scikit-learn pandas matplotlib
uv pip install tensorboard tqdm

# Install audio processing libraries
uv pip install librosa soundfile audioread resampy soxr
uv pip install audiomentations torch-audiomentations

# Install deep learning utilities
uv pip install transformers datasets evaluate huggingface-hub
uv pip install torchinfo

# Optional: Install Jupyter for notebooks
uv pip install ipykernel jupyter
```

#### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import audiomentations; print('Audiomentations OK')"
```

### Alternative: Install from requirements.txt

If you want to use the existing requirements.txt (converted for pip):

```bash
# Install all dependencies
uv pip install -r requirements-uv.txt
```

### Training and Evaluation

Once the environment is set up:

```bash
# Train the evidential model
python train_evidential_model.py
```

#### Reproducing Paper Experiments

**Step 1: Train baseline models (MC Dropout + Deep Ensemble)**

Trains a standard model for MC Dropout (seed=42) and 5 ensemble members (seeds: 42, 123, 456, 789, 1024) using the same hyperparameters as the evidential model (AdamW, lr=1e-4, cosine annealing, 100 epochs, early stopping patience=25).

```bash
python train_baselines.py
```

Saved weights: `weights/standard_mc_dropout_best.pth`, `weights/ensemble_member_{0-4}_best.pth`

**Step 2: UQ method comparison (Table: Evidential vs MC Dropout vs Deep Ensemble)**

Evaluates all three methods on DS1 (in-distribution) and DS3 (domain shift). Reports accuracy, precision, recall, F1, misclassification AUROC, and inference time.

```bash
python evaluate_uq_comparison.py
```

**Step 3: Calibration analysis (ECE, Brier score, reliability diagrams)**

Computes Expected Calibration Error and Brier score for all three methods. Generates reliability diagram PDFs in `figures/`.

```bash
python evaluate_calibration.py
```

Output: `figures/reliability_diagram_ds1.pdf`, `figures/reliability_diagram_ds3.pdf`

**Step 4: Multi-seed stability evaluation**

Trains 5 evidential models with different random seeds and reports mean +/- std for all metrics.

```bash
python evaluate_multi_seed.py
```

**Step 5: Loss function ablation study**

Trains 6 ablation variants (Full, NLL-only, NLL+KL, NLL+Penalty, Full with beta=0.1, Full with beta=1.0) and evaluates each on DS1 and DS3.

```bash
python evaluate_ablation.py
```

**Run all experiments sequentially:**

```bash
python train_baselines.py && \
python evaluate_uq_comparison.py && \
python evaluate_calibration.py && \
python evaluate_multi_seed.py && \
python evaluate_ablation.py
```

Results are saved as `.pth` files in `weights/` and printed as LaTeX-formatted tables to the terminal.

#### Run tests
```bash
python test.py
```

## Citation

If you use something in this repository, please be kind and cite our work through:

```
@article{HOXHA2025139829,
title = {Contrastive learning for robust defect mapping in concrete slabs using impact echo},
journal = {Construction and Building Materials},
volume = {461},
pages = {139829},
year = {2025},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2024.139829},
url = {https://www.sciencedirect.com/science/article/pii/S0950061824049717},
author = {Ejup Hoxha and Jinglun Feng and Agnimitra Sengupta and David Kirakosian and Yang He and Bo Shang and Ardian Gjinofci and Jizhong Xiao},
keywords = {Impact echo, Bridge decks, Contrastive learning, Concrete defects}
}
```
and

```
@ARTICLE{10168232,
  author={Hoxha, Ejup and Feng, Jinglun and Sanakov, Diar and Xiao, Jizhong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Robotic Inspection and Subsurface Defect Mapping Using Impact-Echo and Ground Penetrating Radar}, 
  year={2023},
  volume={8},
  number={8},
  pages={4943-4950},
  doi={10.1109/LRA.2023.3290386}}
```

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Performance Notes

- **MPS (Apple Silicon)**: Automatically detected and used if available
- **CUDA (NVIDIA)**: Install CUDA-specific PyTorch from https://pytorch.org
- **CPU**: Works but slower, consider using smaller batch sizes

## Troubleshooting

### UV is slow or not working
```bash
# Update UV to latest version
uv self update

# Clear UV cache
uv cache clean
```

### PyTorch MPS issues on macOS
```bash
# Ensure you have the latest PyTorch
uv pip install --upgrade torch torchvision torchaudio
```

### Audiomentations installation fails
```bash
# Install system dependencies first (macOS)
brew install libsndfile

# Then retry
uv pip install audiomentations
```

### Import errors
```bash
# Verify environment is activated
which python  # Should show .venv/bin/python

# Reinstall problematic package
uv pip install --reinstall <package-name>
```

## Migrating from Conda

If you're currently using conda (ieenv):

```bash
# Export conda packages (for reference)
conda list --export > conda-packages.txt

# Deactivate conda
conda deactivate

# Create and activate UV environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies as shown above
uv pip install -r requirements-uv.txt
```

## Why UV?

- **10-100x faster** than pip for package installation
- **Deterministic** dependency resolution
- **Compatible** with pip and PyPI
- **Modern** Rust-based implementation
- **Smaller** disk footprint than conda

## Additional Resources

- UV Documentation: https://docs.astral.sh/uv/
- PyTorch Installation: https://pytorch.org/get-started/locally/
- Audiomentations: https://github.com/iver56/audiomentations
