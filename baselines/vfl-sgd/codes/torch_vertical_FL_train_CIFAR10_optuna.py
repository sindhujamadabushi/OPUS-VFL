import optuna
import argparse
import random
import torch
import numpy as np

from torch_vertical_FL_train_CIFAR10 import Vertical_FL_Train, default_organization_num

def objective(trial):
    """
    Optuna objective function to maximize the final validation accuracy 
    from your vertical FL training code.
    """
    # 1) Suggest hyperparameters
    server_lr = trial.suggest_loguniform("server_lr", 1e-4, 1e-2)
    client_lr = trial.suggest_loguniform("client_lr", 1e-4, 1e-2)

    # Possibly tune DP hyperparameters, scheduler gammas, etc.
    dp_noise_multiplier = trial.suggest_float("dp_noise_multiplier", 0.05, 0.5)
    dp_max_grad_norm = trial.suggest_float("dp_max_grad_norm", 0.05, 1.0)

    epochs = 100

    args = argparse.Namespace(
        dname='CIFAR10',
        epochs=epochs,
        batch_type='mini-batch',
        batch_size=64,
        data_type='original',
        model_type='vertical',
        organization_num=default_organization_num,
        poisoning_budget=0.5,
        g_tm=0.0525,   # or tune these with trial
        g_bm=0.0525,
        use_encryption=False
    )

    vfl_model = Vertical_FL_Train(active_clients=None)

    learning_rates = [server_lr, client_lr]
    train_loss_array, val_loss_array, train_auc_array, val_auc_array, _ = vfl_model.run(args, learning_rates, args.batch_size)

    final_val_auc = val_auc_array[-1] if len(val_auc_array) > 0 else 0.0
    return final_val_auc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print("  Value (Val AUC): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")