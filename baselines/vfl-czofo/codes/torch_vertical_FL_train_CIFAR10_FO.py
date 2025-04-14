import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models  # for ResNet
import random
import argparse
import time

###############################################################################
# (We keep the ZO helper here for later use, but it won’t be used in this FO version)
###############################################################################
class ZOOutputOptim:
    """
    Zero-order gradient estimator for an output layer.
    (Not used in pure FO mode)
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
    Return a Subset with samples_per_class examples per class.
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
parser.add_argument('--framework_type', type=str, default="CZOFO")
parser.add_argument('--dataset_name', type=str, default="CIFAR10")
parser.add_argument('--model_type', type=str, default="SimpleResNet18")
parser.add_argument('--n_party', type=int, default=2)
parser.add_argument('--client_output_size', type=int, default=10)
parser.add_argument('--server_embedding_size', type=int, default=128)
# For pure FO, we update all parameters, so we use client_lr as-is.
parser.add_argument('--client_lr', type=float, default=0.01/10)
parser.add_argument('--server_lr', type=float, default=0.01/10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=1000)
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
    Replicate 'dataset' for each party.
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
    models[1..n_party-1]: client models with a ResNet backbone + a linear output.
    """

    class ServerModel(nn.Module):
        def __init__(self, input_dim=client_output_size*(n_party-1), num_classes=10):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    # -----------------------------------------------------------
    #  A real ResNet-based client model
    # -----------------------------------------------------------
    from torchvision.models import resnet18, ResNet18_Weights

    class ClientResNetModel(nn.Module):
        def __init__(self, out_dim=10, pretrained=False):
            super().__init__()
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            backbone.fc = nn.Identity()  # remove the original FC layer
            self.backbone = backbone
            self.client_output_layer = nn.Linear(512, out_dim)

        def forward(self, x):
            feats = self.backbone(x)  # shape [batch, 512]
            out = self.client_output_layer(feats)  # shape [batch, out_dim]
            return out

    # Build the server model:
    models = []
    server_in_dim = (n_party - 1) * client_output_size
    server_model = ServerModel(input_dim=server_in_dim).to(device)
    models.append(server_model)

    # Build client models with the ResNet backbone:
    for i in range(1, n_party):
        cm = ClientResNetModel(out_dim=client_output_size, pretrained=False).to(device)
        models.append(cm)

    return models

def init_optimizers(models, server_lr, client_lr):
    """
    Pure FO optimizers for all parameters (both backbone and final layer).
    """
    opts = []
    # Server optimizer (FO on all parameters)
    server_optimizer = torch.optim.Adam(models[0].parameters(), lr=server_lr)
    opts.append(server_optimizer)

    # Client optimizers (FO on all parameters)
    for m in range(1, len(models)):
        client_optimizer = torch.optim.Adam(models[m].parameters(), lr=client_lr)
        opts.append(client_optimizer)

    return opts

def append_log(log_item, log_file):
    pass

###############################################################################
# Main training function (Pure FO updates)
###############################################################################
def main():
    # Hyperparameters
    client_lr = args.client_lr
    server_lr = args.server_lr
    n_epoch_trial = run_epochs

    # Re-seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load dataset
    trainset_full, testset_full = get_dataset(args.dataset_name)
    # For debugging, use a small subset:
    trainset = make_balanced_subset(trainset_full, samples_per_class=2500)  # 50*10 = 500 samples
    testset = make_balanced_subset(testset_full, samples_per_class=1000)     # 10*10 = 100 samples

    # Partition dataset
    n_party = args.n_party
    batch_size = args.batch_size
    train_dataset_list, train_loader_list = partition_dataset(args.dataset_name, trainset, n_party, batch_size, shuffle=True)
    pre_extracted_batches = [list(dl) for dl in train_loader_list]
    n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

    # Fresh loaders for evaluation
    train_eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_eval_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Build objects
    loss_fn = nn.CrossEntropyLoss()
    # (We keep ZOO here for later; not used in pure FO)
    ZOO = ZOOutputOptim(args.mu, args.u_type, args.client_output_size, n_directions=args.sample_times)
    class DummyCompressor:
        def compress_decompress(self, x):
            return x
    compressor = DummyCompressor()
    response_compressor = DummyCompressor()

    # Build models + optimizers
    models = init_models(args.model_type, n_party, train_dataset_list, device, args.client_output_size, args.server_embedding_size)
    optimizers = init_optimizers(models, server_lr, client_lr)

    print("Starting pure FO training...")
    for epoch in range(n_epoch_trial):
        # Training loop (pure FO, so update entire client model normally)
        for batch_idx in range(n_training_batch_per_epoch_client):
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            labels_for_loss = None

            for m in range(1, n_party):
                inp_m, lab_m = pre_extracted_batches[m][batch_idx]
                inp_m = inp_m.to(device)
                lab_m = lab_m.to(device)
                if m == 1:
                    labels_for_loss = lab_m

                # Entire forward pass (backbone and final layer) with gradients
                out_full = models[m](inp_m)
                out_emb_dict[m] = out_full

            server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
            server_pred = models[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()

            for opt in optimizers:
                opt.step()

        # ------------------------------
        # End-of-epoch: Evaluate training and test accuracy
        # ------------------------------
        models[0].eval()
        for m in range(1, n_party):
            models[m].eval()

        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for inp, lbl in train_eval_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = []
                for m in range(1, n_party):
                    out_list.append(models[m](inp))
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models[0](server_in_eval)
                _, pred = torch.max(out_server, 1)
                train_total += lbl.size(0)
                train_correct += (pred == lbl).sum().item()
        train_acc = 100.0 * train_correct / train_total if train_total else 0

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inp, lbl in test_eval_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out_list = []
                for m in range(1, n_party):
                    out_list.append(models[m](inp))
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models[0](server_in_eval)
                _, predicted = torch.max(out_server, 1)
                test_total += lbl.size(0)
                test_correct += (predicted == lbl).sum().item()
        test_acc = 100.0 * test_correct / test_total if test_total else 0

        print(f"Epoch [{epoch}/{n_epoch_trial}]  Pure FO Train Acc = {train_acc:.2f}%   Test Acc = {test_acc:.2f}%")

        models[0].train()
        for m in range(1, n_party):
            models[m].train()

    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()