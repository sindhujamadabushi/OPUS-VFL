import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import argparse
import optuna
import time

###############################################################################
# Zero-Order helper
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
        """
        Return lists of (u, out_plus, out_minus) for multiple random directions.
        """
        U = []
        plus_list = []
        minus_list = []
        for _ in range(self.n_directions):
            if self.u_type == "Uniform":
                u = torch.randn_like(original_out)
            else:
                u = torch.randn_like(original_out)
            out_plus = original_out + self.mu * u
            out_minus = original_out - self.mu * u
            U.append(u)
            plus_list.append(out_plus)
            minus_list.append(out_minus)
        return U, plus_list, minus_list

    def backward(self, U, loss_diffs):
        """
        Compute the estimated gradient averaged over all directions.
        For each direction:
           grad_i = (loss_plus_i - loss_minus_i) / (2 * mu) * U[i]
        """
        grad_accum = 0.0
        for i in range(self.n_directions):
            grad_i = (loss_diffs[i] / (2.0 * self.mu)) * U[i]
            grad_accum += grad_i
        grad_est = grad_accum / float(self.n_directions)
        return grad_est

###############################################################################
# Balanced subset helper for CIFAR-10
###############################################################################
def make_balanced_subset(dataset, samples_per_class):
    """
    Given a dataset with 10 classes, return a Subset with samples_per_class examples per class.
    """
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
# (Example) compress_decompress utility
###############################################################################
def compress_decompress(tensor, ctype, compressor):
    if ctype == "None":
        return tensor
    return compressor.compress_decompress(tensor)

###############################################################################
# Parse command-line arguments
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12341)
parser.add_argument('--framework_type', type=str, default="ZO")
parser.add_argument('--dataset_name', type=str, default="CIFAR10")
parser.add_argument('--model_type', type=str, default="SimpleResNet18")
parser.add_argument('--n_party', type=int, default=3)
parser.add_argument('--client_output_size', type=int, default=10)
parser.add_argument('--server_embedding_size', type=int, default=128)
parser.add_argument('--client_lr', type=float, default=0.002)
parser.add_argument('--server_lr', type=float, default=0.002)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--u_type', type=str, default="Uniform")
parser.add_argument('--mu', type=float, default=0.001)
parser.add_argument('--d', type=float, default=1)
parser.add_argument('--sample_times', type=int, default=20)
parser.add_argument('--compression_type', type=str, default="None")
parser.add_argument('--compression_bit', type=int, default=8)
parser.add_argument('--response_compression_type', type=str, default="None")
parser.add_argument('--response_compression_bit', type=int, default=8)
parser.add_argument('--local_update_times', type=int, default=1)
parser.add_argument('--log_file_name', type=str, default="None")
parser.add_argument('--attack_type', type=str, default=None)
parser.add_argument('--severity', type=float, default=0.1)
args = parser.parse_args()

run_epochs = args.n_epoch
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################################
# Dataset and Partition Helpers
###############################################################################
def get_dataset(name):
    """
    Returns train_dataset and test_dataset with CIFAR-10 transformations.
    """
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
    """
    Partition 'dataset' into n_party subsets. In VFL, each party sees the same indices.
    For simplicity, we return the same dataset and create a DataLoader with shuffling.
    """
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
    """
    Build server and client models.
    models[0]: server model.
    models[1..n_party-1]: client models with a backbone and a client_output_layer.
    """
    class ServerModel(nn.Module):
        def __init__(self, input_dim=client_output_size*(n_party-1), num_classes=10):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    class ClientModel(nn.Module):
        def __init__(self, backbone_in_dim=32*32*3, hidden_dim=512, out_dim=10):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(backbone_in_dim, hidden_dim),
                nn.ReLU()
            )
            self.client_output_layer = nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            feats = self.backbone(x)
            out = self.client_output_layer(feats)
            return out

    models = []
    server_in_dim = (n_party - 1) * client_output_size
    server_model = ServerModel(input_dim=server_in_dim).to(device)
    models.append(server_model)
    for i in range(1, n_party):
        cm = ClientModel().to(device)
        models.append(cm)
    return models

def init_optimizers(models, server_lr, client_lr):
    """
    Initialize optimizers.
    - Server: FO on all parameters.
    - Clients: FO on backbone parameters only (exclude client_output_layer, which will be updated via ZO).
    """
    opts = []
    # Server optimizer (FO on entire network)
    server_params = list(models[0].parameters())
    server_optimizer = torch.optim.Adam(server_params, lr=server_lr)
    opts.append(server_optimizer)
    # Client optimizers
    for m in range(1, len(models)):
        backbone_params = []
        final_layer_params = []
        for name, param in models[m].named_parameters():
            if "client_output_layer" in name:
                final_layer_params.append(param)
            else:
                backbone_params.append(param)
        # Disable FO for final layer (will be updated manually via ZO)
        for p in final_layer_params:
            p.requires_grad = False
        client_optimizer = torch.optim.Adam(backbone_params, lr=client_lr)
        opts.append(client_optimizer)
    return opts

def append_log(log_item, log_file):
    # Dummy log appending function
    pass

###############################################################################
# Main objective function for Optuna
###############################################################################
def objective(trial):
    # Hyperparameters from Optuna
    client_lr = trial.suggest_loguniform("client_lr", 1e-4, 1e-2)
    server_lr = trial.suggest_loguniform("server_lr", 1e-4, 1e-2)
    mu = trial.suggest_loguniform("mu", 1e-4, 1e-2)
    n_epoch_trial = run_epochs

    # Re-seed for reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load datasets and create balanced subsets
    trainset_full, testset_full = get_dataset(args.dataset_name)
    trainset = make_balanced_subset(trainset_full, samples_per_class=25000)
    testset = make_balanced_subset(testset_full, samples_per_class=5000)

    # Partition training dataset for VFL (shuffle=True)
    n_party = args.n_party
    batch_size = args.batch_size
    train_dataset_list, train_loader_list = partition_dataset(
        args.dataset_name, trainset, n_party, batch_size, shuffle=True
    )
    pre_extracted_batches = [list(dl) for dl in train_loader_list]
    n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

    # Build loss, ZO object and compressor placeholders
    loss_fn = nn.CrossEntropyLoss()
    ZOO = ZOOutputOptim(mu, args.u_type, args.client_output_size, n_directions=args.sample_times)
    # Dummy compressor (if not used)
    class DummyCompressor:
        def compress_decompress(self, x):
            return x
    compressor = DummyCompressor()
    response_compressor = DummyCompressor()

    # Build models and optimizers
    models = init_models(args.model_type, n_party, train_dataset_list, device, 
                         args.client_output_size, args.server_embedding_size)
    optimizers = init_optimizers(models, server_lr, client_lr)

    # Training loop
    for epoch in range(n_epoch_trial):
        for batch_idx in range(n_training_batch_per_epoch_client):
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            hidden_feats = {}
            labels_for_loss = None

            # Forward pass for each client (clients 1..n_party-1)
            for m in range(1, n_party):
                # Unpack only input and label since CIFAR10 returns 2 items
                inp_m, lab_m = pre_extracted_batches[m][batch_idx]
                inp_m = inp_m.to(device)
                lab_m = lab_m.to(device)
                if m == 1:
                    labels_for_loss = lab_m
                # Get backbone features (FO)
                feats = models[m].backbone(inp_m)
                hidden_feats[m] = feats.detach()  # save features for ZO update
                # Compute final output (ZO final layer; no FO gradient)
                with torch.no_grad():
                    out_full = models[m].client_output_layer(feats)
                out_emb = compress_decompress(out_full, args.compression_type, compressor)
                out_emb_dict[m] = out_emb

            # Server forward pass (FO update)
            server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
            server_pred = models[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()
            # Update server and client backbones
            for opt in optimizers:
                opt.step()

            # Zero-Order update for each client's final layer
            for m in range(1, n_party):
                out_original = out_emb_dict[m].detach()
                # Get multiple directions
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
                        server_pred_plus = models[0](server_in_plus)
                        server_pred_minus = models[0](server_in_minus)
                        loss_plus = loss_fn(server_pred_plus, labels_for_loss)
                        loss_minus = loss_fn(server_pred_minus, labels_for_loss)
                    loss_diff = loss_plus - loss_minus
                    loss_diffs.append(loss_diff)
                # Estimate gradient with respect to final layer's output
                est_grad_out = ZOO.backward(U, loss_diffs)
                # Manually update final layer parameters (only ZO update)
                feats = hidden_feats[m].to(device)  # shape: [batch, hidden_dim]
                dWeight = torch.matmul(est_grad_out.transpose(0, 1), feats)
                dBias = est_grad_out.sum(dim=0)
                with torch.no_grad():
                    models[m].client_output_layer.weight -= client_lr * dWeight
                    models[m].client_output_layer.bias   -= client_lr * dBias

        # ----- End of epoch: Evaluate training accuracy -----
        # ----- End of epoch: Evaluate training accuracy -----
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            # Iterate over aligned batches from all parties using zip
            for batches in zip(*pre_extracted_batches):
                # Each batch is a tuple from each party; CIFAR-10 returns (inp, label)
                inp0, lbl0 = batches[0]
                inp0 = inp0.to(device)
                lbl0 = lbl0.to(device)
                out_eval_list = []
                for m in range(1, n_party):
                    inp_m, _ = batches[m]
                    inp_m = inp_m.to(device)
                    out_eval = models[m](inp_m)
                    out_eval_list.append(out_eval)
                server_in_eval = torch.cat(out_eval_list, dim=-1)
                out_server = models[0](server_in_eval)
                _, pred = torch.max(out_server, 1)
                train_total += lbl0.size(0)
                train_correct += (pred == lbl0).sum().item()
        train_acc = 100.0 * train_correct / train_total if train_total else 0

        # ----- End of epoch: Evaluate test accuracy -----
        test_dataset_list, test_loader_list = partition_dataset(
            args.dataset_name, testset, n_party, batch_size, shuffle=False
        )
        correct = 0
        total = 0
        with torch.no_grad():
            # Align test batches across parties
            for test_batches in zip(*test_loader_list):
                inp0, lbl0 = test_batches[0]
                inp0 = inp0.to(device)
                lbl0 = lbl0.to(device)
                outputs_list = [None] * n_party
                for m in range(1, n_party):
                    inp_m, _ = test_batches[m]
                    inp_m = inp_m.to(device)
                    outputs_list[m] = models[m](inp_m)
                server_in = torch.cat(outputs_list[1:], dim=-1)
                outputs = models[0](server_in)
                _, predicted = torch.max(outputs, 1)
                total += lbl0.size(0)
                correct += (predicted == lbl0).sum().item()
        test_acc = 100.0 * correct / total if total else 0

        print(f"Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")
        append_log([0, 0, 0, test_acc, 0], args.log_file_name)
    return test_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")