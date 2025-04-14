#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models  # for ResNet
import random
import argparse
import time
import numpy as np

def create_backdoor_trainset(dataset, source_class, trigger_value, window_size):
    bd_inputs = []
    bd_labels = []
    for img, label in dataset:
        if label == source_class:
            img_triggered = insert_trigger_patch(img.clone(), 5, 5, window_size, trigger_value)
            bd_inputs.append(img_triggered)
            bd_labels.append(label)
    return torch.stack(bd_inputs), torch.tensor(bd_labels)

def create_backdoor_testset(dataset, source_class, trigger_value, window_size):
    bd_inputs = []
    bd_labels = []
    for img, label in dataset:
        if label == source_class:
            img_triggered = insert_trigger_patch(img.clone(), 5, 5, window_size, trigger_value)
            bd_inputs.append(img_triggered)
            bd_labels.append(label)
    return torch.stack(bd_inputs), torch.tensor(bd_labels)

###############################################################################
# Zero-Order Helper
###############################################################################
class ZOOutputOptim:
    """
    Zero-order gradient estimator for an output layer.
    """
    def __init__(self, mu, u_type, output_dim, n_directions=1):
        self.mu = mu
        self.u_type = u_type
        self.output_dim = output_dim
        self.n_directions = n_directions

    def forward_example(self, original_out):
        U = []
        plus_list = []
        minus_list = []
        for _ in range(self.n_directions):
            # Using a standard Gaussian perturbation.
            u = torch.randn_like(original_out)
            out_plus = original_out + self.mu * u
            out_minus = original_out - self.mu * u
            U.append(u)
            plus_list.append(out_plus)
            minus_list.append(out_minus)
        return U, plus_list, minus_list

    def backward(self, U, loss_diffs):
        grad_accum = 0.0
        for i in range(self.n_directions):
            grad_i = (loss_diffs[i] / (2.0 * self.mu)) * U[i]
            grad_accum += grad_i
        grad_est = grad_accum / float(self.n_directions)
        return grad_est

###############################################################################
# Balanced Subset Helper for CIFAR-10
###############################################################################
def make_balanced_subset(dataset, samples_per_class):
    class_indices = [[] for _ in range(10)]
    for idx, (data, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    final_indices = []
    for c in range(10):
        random.shuffle(class_indices[c])
        final_indices.extend(class_indices[c][:samples_per_class])
    return Subset(dataset, final_indices)

###############################################################################
# Backdoor Attack Helpers
###############################################################################
def find_best_patch_location(saliency_map, window_size):
    return 5, 5

def insert_trigger_patch(img_tensor, r0, c0, window_size, trigger_value):
    img_tensor[:, r0:r0+window_size, c0:c0+window_size] = trigger_value
    return img_tensor

###############################################################################
# Argument Parsing
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12341)
parser.add_argument('--framework_type', type=str, default="CZOFO")
parser.add_argument('--dataset_name', type=str, default="CIFAR10")
parser.add_argument('--model_type', type=str, default="SimpleResNet18")
parser.add_argument('--n_party', type=int, default=5)
parser.add_argument('--client_output_size', type=int, default=10)
parser.add_argument('--server_embedding_size', type=int, default=128)
parser.add_argument('--client_lr', type=float, default=0.01/10)
parser.add_argument('--server_lr', type=float, default=0.01/10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--u_type', type=str, default="Uniform")
parser.add_argument('--mu', type=float, default=0.001)
parser.add_argument('--d', type=float, default=1)
parser.add_argument('--sample_times', type=int, default=5)
parser.add_argument('--compression_type', type=str, default="None")
parser.add_argument('--compression_bit', type=int, default=8)
parser.add_argument('--response_compression_type', type=str, default="None")
parser.add_argument('--response_compression_bit', type=int, default=8)
parser.add_argument('--local_update_times', type=int, default=1)
parser.add_argument('--log_file_name', type=str, default="None")
# Backdoor-specific arguments:
parser.add_argument('--attack_type', type=str, default="backdoor")
parser.add_argument('--severity', type=float, default=0.1)
parser.add_argument('--poisoning_budget', type=float, default=0.1)
args = parser.parse_args()

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Dataset and Partition Helpers
###############################################################################
def get_dataset(name):
    if name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        trainset_full = datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=transform_train)
        testset_full = datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform_test)
        return trainset_full, testset_full
    return None, None

def partition_dataset(name, dataset, n_party, batch_size, shuffle=True):
    dataset_list = []
    loaders = []
    for i in range(n_party):
        dataset_list.append(dataset)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        loaders.append(dl)
    return dataset_list, loaders

###############################################################################
# Model and Optimizer Initialization
###############################################################################
def init_models(model_type, n_party, train_dataset_list, device, client_output_size, server_embedding_size):
    class ServerModel(nn.Module):
        def __init__(self, input_dim=client_output_size*(n_party-1), num_classes=10):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    from torchvision.models import resnet18, ResNet18_Weights

    class ClientResNetModel(nn.Module):
        def __init__(self, out_dim=10, pretrained=False):
            super().__init__()
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.client_output_layer = nn.Linear(512, out_dim)
        def forward(self, x):
            feats = self.backbone(x)
            out = self.client_output_layer(feats)
            return out

    models_list = []
    server_in_dim = (n_party - 1) * client_output_size
    server_model = ServerModel(input_dim=server_in_dim).to(device)
    models_list.append(server_model)
    for i in range(1, n_party):
        cm = ClientResNetModel(out_dim=client_output_size, pretrained=False).to(device)
        models_list.append(cm)
    return models_list

def init_optimizers(models_list, server_lr, client_lr):
    opts = []
    server_optimizer = torch.optim.Adam(models_list[0].parameters(), lr=server_lr)
    opts.append(server_optimizer)
    for m in range(1, len(models_list)):
        client_optimizer = torch.optim.Adam(models_list[m].parameters(), lr=client_lr)
        opts.append(client_optimizer)
    return opts

###############################################################################
# Training and Evaluation Function
###############################################################################
def run_experiment():
    # Hyperparameters
    client_lr = args.client_lr
    server_lr = args.server_lr
    mu = args.mu
    n_epoch_trial = args.n_epoch

    # Set up dataset
    trainset_full, testset_full = get_dataset(args.dataset_name)
    trainset = make_balanced_subset(trainset_full, samples_per_class=2500)
    testset = make_balanced_subset(testset_full, samples_per_class=1000)

    n_party = args.n_party
    batch_size = args.batch_size
    train_dataset_list, train_loader_list = partition_dataset(args.dataset_name, trainset, n_party, batch_size, shuffle=True)
    pre_extracted_batches = [list(dl) for dl in train_loader_list]
    n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

    train_eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_eval_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    ZOO = ZOOutputOptim(mu, args.u_type, args.client_output_size, n_directions=args.sample_times)

    models_list = init_models(args.model_type, n_party, train_dataset_list, device, args.client_output_size, args.server_embedding_size)
    optimizers = init_optimizers(models_list, server_lr, client_lr)

    # Backdoor parameters
    malicious_client_idx = 1
    source_class = 3
    target_class = 8
    trigger_value = 0.02
    window_size = 5
    poisoning_budget = args.poisoning_budget

    # Inject backdoor into malicious client's data
    for batch_idx in range(n_training_batch_per_epoch_client):
        inp, lab = pre_extracted_batches[malicious_client_idx][batch_idx]
        inp_cpu = inp.cpu()
        lab_cpu = lab.cpu()
        for i in range(inp_cpu.shape[0]):
            if lab_cpu[i].item() == source_class and random.random() < poisoning_budget:
                inp_cpu[i] = insert_trigger_patch(inp_cpu[i].clone(), 5, 5, window_size, trigger_value)
                lab_cpu[i] = target_class
        pre_extracted_batches[malicious_client_idx][batch_idx] = (inp_cpu.to(device), lab_cpu.to(device))

    # To record metrics across epochs
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_accs = []
    epoch_test_accs = []
    epoch_train_asrs = []
    epoch_test_asrs = []

    print("Starting training with FO + ZO updates and integrated backdoor attack...")

    for epoch in range(n_epoch_trial):
        running_loss = 0.0
        st = time.time()
        for batch_idx in range(n_training_batch_per_epoch_client):
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            hidden_feats = {}
            labels_for_loss = None

            # Forward pass for each client (from index 1 onward)
            for m in range(1, n_party):
                inp_m, lab_m = pre_extracted_batches[m][batch_idx]
                inp_m = inp_m.to(device)
                lab_m = lab_m.to(device)
                if m == 1:
                    labels_for_loss = lab_m  # malicious client's labels (poisoned)
                feats = models_list[m].backbone(inp_m)
                hidden_feats[m] = feats.detach()
                out_full = models_list[m].client_output_layer(feats)
                out_emb_dict[m] = out_full

            server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
            server_pred = models_list[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()
            running_loss += loss.item()

            for opt in optimizers:
                opt.step()

            # ZO update for client final layers
            for m in range(1, n_party):
                with torch.no_grad():
                    out_original = models_list[m].client_output_layer(hidden_feats[m])
                U, plus_list, minus_list = ZOO.forward_example(out_original)
                loss_diffs = []
                for d_i in range(ZOO.n_directions):
                    plus_emb_dict = dict(out_emb_dict)
                    minus_emb_dict = dict(out_emb_dict)
                    plus_emb_dict[m] = plus_list[d_i]
                    minus_emb_dict[m] = minus_list[d_i]
                    server_in_plus = torch.cat([plus_emb_dict[x] for x in range(1, n_party)], dim=-1)
                    server_in_minus = torch.cat([minus_emb_dict[x] for x in range(1, n_party)], dim=-1)
                    with torch.no_grad():
                        server_pred_plus = models_list[0](server_in_plus)
                        server_pred_minus = models_list[0](server_in_minus)
                        loss_plus = loss_fn(server_pred_plus, labels_for_loss)
                        loss_minus = loss_fn(server_pred_minus, labels_for_loss)
                    loss_diff = loss_plus - loss_minus
                    loss_diffs.append(loss_diff)
                est_grad_out = ZOO.backward(U, loss_diffs)
                feats = hidden_feats[m].to(device)
                dWeight = torch.matmul(est_grad_out.transpose(0, 1), feats)
                dBias = est_grad_out.sum(dim=0)
                with torch.no_grad():
                    models_list[m].client_output_layer.weight -= client_lr * dWeight
                    models_list[m].client_output_layer.bias   -= client_lr * dBias

        # Create a backdoor train loader for measuring training ASR
        bd_train_inputs, bd_train_labels = create_backdoor_trainset(trainset, source_class, trigger_value, window_size)
        bd_train_loader = DataLoader(list(zip(bd_train_inputs, bd_train_labels)), batch_size=batch_size, shuffle=False)
        # Create a backdoor test loader for measuring test ASR
        bd_test_inputs, bd_test_labels = create_backdoor_testset(testset, source_class, trigger_value, window_size)
        bd_test_loader = DataLoader(list(zip(bd_test_inputs, bd_test_labels)), batch_size=batch_size, shuffle=False)

        # Evaluation on clean train set
        models_list[0].eval()
        for m in range(1, n_party):
            models_list[m].eval()
        train_correct = 0
        train_total = 0
        total_train_loss = 0.0
        with torch.no_grad():
            for inp, lbl in train_eval_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = [models_list[m](inp) for m in range(1, n_party)]
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models_list[0](server_in_eval)
                loss_batch = loss_fn(out_server, lbl)
                total_train_loss += loss_batch.item() * lbl.size(0)
                _, pred = torch.max(out_server, 1)
                train_total += lbl.size(0)
                train_correct += (pred == lbl).sum().item()
        train_acc = 100.0 * train_correct / train_total if train_total else 0
        avg_train_loss = total_train_loss / train_total

        # Evaluation on clean test set
        test_correct = 0
        test_total = 0
        total_test_loss = 0.0
        with torch.no_grad():
            for inp, lbl in test_eval_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = [models_list[m](inp) for m in range(1, n_party)]
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models_list[0](server_in_eval)
                loss_batch = loss_fn(out_server, lbl)
                total_test_loss += loss_batch.item() * lbl.size(0)
                _, predicted = torch.max(out_server, 1)
                test_total += lbl.size(0)
                test_correct += (predicted == lbl).sum().item()
        test_acc = 100.0 * test_correct / test_total if test_total else 0
        avg_test_loss = total_test_loss / test_total

        print("time taken to run 1 epoch: ", time.time()-st)
        # Measure Training ASR on backdoor train set
        train_asr_total = 0
        train_asr_success = 0
        with torch.no_grad():
            for inp, lbl in bd_train_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = [models_list[m](inp) for m in range(1, n_party)]
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models_list[0](server_in_eval)
                _, pred = torch.max(out_server, 1)
                train_asr_total += lbl.size(0)
                train_asr_success += (pred == target_class).sum().item()
        train_asr = 100.0 * train_asr_success / train_asr_total if train_asr_total else 0

        # Measure Test ASR on backdoor test set
        test_asr_total = 0
        test_asr_success = 0
        with torch.no_grad():
            for inp, lbl in bd_test_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = [models_list[m](inp) for m in range(1, n_party)]
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models_list[0](server_in_eval)
                _, pred = torch.max(out_server, 1)
                test_asr_total += lbl.size(0)
                test_asr_success += (pred == target_class).sum().item()
        test_asr = 100.0 * test_asr_success / test_asr_total if test_asr_total else 0

        print(f"Epoch [{epoch}/{n_epoch_trial}] Train Acc = {train_acc:.2f}%, Train Loss = {avg_train_loss:.4f}, Test Acc = {test_acc:.2f}%, Test Loss = {avg_test_loss:.4f}")
        print(f"           Train ASR = {train_asr:.2f}%, Test ASR = {test_asr:.2f}%")

        epoch_train_accs.append(train_acc)
        epoch_test_accs.append(test_acc)
        epoch_train_losses.append(avg_train_loss)
        epoch_test_losses.append(avg_test_loss)
        epoch_train_asrs.append(train_asr)
        epoch_test_asrs.append(test_asr)

        # Set models back to training mode for the next epoch
        models_list[0].train()
        for m in range(1, n_party):
            models_list[m].train()

    # Return final epoch metrics (you can also average across epochs if desired)
    return {
        'train_acc': epoch_train_accs[-1],
        'test_acc': epoch_test_accs[-1],
        'train_loss': epoch_train_losses[-1],
        'test_loss': epoch_test_losses[-1],
        'train_asr': epoch_train_asrs[-1],
        'test_asr': epoch_test_asrs[-1]
    }

if __name__ == "__main__":
    n_runs = 10
    all_metrics = []
    for run in range(n_runs):
        print(f"\n=== Starting Experiment Run {run+1}/{n_runs} ===")
        metrics = run_experiment()
        all_metrics.append(metrics)
    
    # Compute average metrics over runs
    avg_train_acc  = np.mean([m['train_acc'] for m in all_metrics])
    avg_test_acc   = np.mean([m['test_acc'] for m in all_metrics])
    avg_train_loss = np.mean([m['train_loss'] for m in all_metrics])
    avg_test_loss  = np.mean([m['test_loss'] for m in all_metrics])
    avg_train_asr  = np.mean([m['train_asr'] for m in all_metrics])
    avg_test_asr   = np.mean([m['test_asr'] for m in all_metrics])
    
    print("\n=== Average Metrics Over 10 Runs ===")
    print(f"Train Acc: {avg_train_acc:.2f}%")
    print(f"Test Acc: {avg_test_acc:.2f}%")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Train ASR: {avg_train_asr:.2f}%")
    print(f"Test ASR: {avg_test_asr:.2f}%")