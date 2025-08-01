## Installation

```bash
pip install -r requirements.txt
```

## Running the Code

All runnable scripts are located in the `baselines/opus-vfl/codes/` folder.  
We provide two types of scripts:

- **High-level runners**: Files starting with `run_` — use these to directly launch experiments.
- **Core training scripts**: Files starting with `torch_vertical_FL_train_incentive_` — these contain the main training loop and logic, called internally by the `run_` scripts.

### Quick Start

Use the appropriate `run_*.py` file based on the setting you want to test:

| Task                      | Dataset   | Script to Run                               |
|---------------------------|-----------|----------------------------------------------|
| Clean training            | MNIST     | `run_p2vfl_mnist.py`                         |
| Clean training            | CIFAR-10  | `run_p2vfl_cifar10.py`                       |
| Clean training            | CIFAR-100 | `run_p2vfl_cifar100.py`                      |
| Backdoor Attack           | CIFAR-10  | `run_p2vfl_backdoor_cifar10.py`             |
| Feature Inference Attack  | CIFAR-10  | `run_p2vfl_fia_cifar10.py`                  |
| Label Inference Attack    | CIFAR-10  | `run_p2vfl_lia_cifar10.py`                  |
| Label Inference Attack    | MNIST     | `run_p2vfl_lia_mnist.py`                    |

Each script runs end-to-end, including data loading, model training, attack (if applicable), and logging results.

### Other Baselines

We follow the same naming and usage convention for other baselines (e.g., `vanilla`, `vfl_czofo`, `bpf`, `sdg`, etc.).  
Just run the appropriate `run_<baseline>_<attack>_<dataset>.py` script, and it will launch the corresponding experiment.

> **Tip**: All runnable scripts start with `run_`. You can identify them by name to know exactly which setup they refer to.

## Hyperparameters

Each experimental setting in OPUS-VFL uses its own set of tuned hyperparameters.  
Below are the default values used for each setting:

| Setting                        | Dataset   | Top LR       | Org LR       | Batch Size |
|-------------------------------|-----------|--------------|--------------|------------|
| Clean Training                | CIFAR-10  | 0.0005       | 0.0005       | 512        |
| Clean Training                | CIFAR-100 | 0.01         | 0.01         | 512        |
| Clean Training                | MNIST     | 0.0000116    | 0.0000058    | 512        |
| Backdoor Attack               | CIFAR-10  | 0.0005       | 0.0005       | 512        |
| Feature Inference Attack      | CIFAR-10  | 0.0005       | 0.0005       | 512        |
| Label Inference Attack        | CIFAR-10  | 0.0005       | 0.0005       | 512        |
| Label Inference Attack        | MNIST     | 0.0000116    | 0.0000058    | 512        |
 
> - `Top LR` = Learning rate for the server/top model  
> - `Org LR` = Learning rate for organization/client models

> **Note:**  
> The hyperparameters listed above are **directly defined inside each `run_*.py` script** located in `baselines/opus-vfl/codes/`.  
> For example, the hyperparameters for the backdoor attack on CIFAR-10 are set inside `run_p2vfl_backdoor_cifar10.py`.

> **Other Baselines:**  
> If you're running baselines like `vanilla`, `vfl_czofo`, `bpf`, or `sdg`, please check their corresponding `run_*.py` files in the same folder.  
> Each baseline defines its own learning rates, batch size, and other settings **at the top of the script**.

### Hyperparameter Tuning Notes

We primarily used manual tuning to select hyperparameters across all settings.  
For a few challenging cases, we used [Optuna](https://optuna.org) to guide the initial search, followed by further manual refinement.

- **Manual Tuning**:
  - Started with a learning rate of **0.01**
  - Gradually reduced it based on model behavior 
  - Final values were chosen when the model showed stable training and reasonable validation performance.

- **Optuna-Aided Tuning** (used selectively):
  - Search space:
    - Top model LR: 1e-2 to 1e-6
    - Org model LR: 1e-2 to 1e-6
    - Batch size: {64, 128, 256, 512, 1024}
  - Objective: Maximize validation accuracy
  - Trials: 100 per setting
  - Resulting values were further manually adjusted if needed

All final hyperparameters are hardcoded in the corresponding `run_*.py` scripts for full reproducibility.

### Randomness and Reproducibility

We used fixed random seeds during initial tuning and development while selecting stable hyperparameters.  
Specifically, we set seeds for:

- Python `random`
- NumPy

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```
Once stable settings were identified, we disabled the seed and ran each experiment 10 times to capture variability and report averaged results across runs.

### Computing Infrastructure

We used computing clusters for all experiments. The specific hardware varied by dataset:

#### MNIST (CPU-only)
- **Processor**: AMD EPYC 7702 (128 cores/node)
- **Memory**: 256 GB RAM
- **GPU**: Not used
- **Usage**: MNIST experiments were run entirely on CPU nodes.

#### CIFAR-10 & CIFAR-100 (GPU-enabled)
- **GPU**: NVIDIA A30 (24 GB)
- **CPU**: Intel Xeon Platinum 8462Y+ (64 cores/node)
- **Memory**: 512 GB RAM
- **Usage**: All CIFAR-10 and CIFAR-100 experiments were run on GPU nodes with A30 GPUs.

#### Software Environment (All Experiments)
- **Operating System**: Red Hat Enterprise Linux
- **Python**: 3.9+
- **PyTorch**: 1.13+
- **CUDA**: 11.6 (for GPU runs)
- **Other Libraries**: NumPy, scikit-learn, Optuna, tqdm  
  *(Full list available in `requirements.txt`)*

All experiments were run using isolated environments configured via clusters available in our organization's module system. 

## Acknowledgment

Our OPUS-VFL implementation builds upon the original codebase developed by:

- Kang Wei  
- Jun Li  
- Chuan Ma  
- Ming Ding  
- Sha Wei  
- Fan Wu  
- Guihai Chen  
- Thilina Ranbaduge  

Their work is described in the paper: [arXiv:2202.04309](https://arxiv.org/abs/2202.04309)  
Original GitHub repository: [github.com/AdamWei-boop/Vertical_FL](https://github.com/AdamWei-boop/Vertical_FL)

We have adapted their implementation and extended it with additional modules and experiments to support our proposed methods.





