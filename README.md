## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
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

## List of hyperparameters ##


