import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models  # <-- for ResNet
import random
import argparse
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
        For each direction i:
           grad_i = (loss_plus_i - loss_minus_i) / (2 * mu) * U[i]
        Then average across n_directions.
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
    Partition 'dataset' into n_party subsets. In standard 'vertical' FL,
    each party sees the same examples but different features. Here,
    we replicate 'dataset' for each party.
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
            # Load resnet18 from torchvision:
            # If you want pretrained, set pretrained=True, but that
            # typically expects 3x224x224. For 32x32, it's still workable, but not optimal.
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            # backbone.fc is typically a Linear(512, 1000). We'll remove it
            # and replace it with Identity so we get a 512-dim output.
            backbone.fc = nn.Identity()
            self.backbone = backbone
            # Now define the final layer for your client:
            self.client_output_layer = nn.Linear(512, out_dim)

        def forward(self, x):
            # x is shape [batch, 3, 32, 32]
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
        # If you want pretrained, pass `pretrained=True`:
        cm = ClientResNetModel(out_dim=client_output_size, pretrained=False).to(device)
        models.append(cm)

    return models

def init_optimizers(models, server_lr, client_lr):
    """
    - Server: FO on entire network.
    - Clients: FO on backbone only, final layer updated by ZO => requires_grad=False for final layer.
    """
    opts = []
    # Server optimizer (FO on all parameters)
    server_params = list(models[0].parameters())
    server_optimizer = torch.optim.Adam(server_params, lr=server_lr)
    opts.append(server_optimizer)

    # Client optimizers
    for m in range(1, len(models)):
        # Split backbone vs final layer:
        backbone_params = []
        final_layer_params = []
        for name, param in models[m].named_parameters():
            if "client_output_layer" in name:
                final_layer_params.append(param)
            else:
                backbone_params.append(param)
        # Zero-order for final layer => disable FO
        for p in final_layer_params:
            p.requires_grad = False
        # FO on the backbone
        client_optimizer = torch.optim.Adam(backbone_params, lr=client_lr)
        opts.append(client_optimizer)

    return opts

def append_log(log_item, log_file):
    pass

###############################################################################
# Main training function
###############################################################################
def main():
    # Hyperparameters
    client_lr = args.client_lr
    server_lr = args.server_lr
    mu = args.mu
    n_epoch_trial = run_epochs

    # Re-seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load dataset
    trainset_full, testset_full = get_dataset(args.dataset_name)
    # You can do smaller subsets if you want to debug quickly:
    # trainset = make_balanced_subset(trainset_full, samples_per_class=5000)
    # testset  = make_balanced_subset(testset_full,  samples_per_class=1000)

    trainset = make_balanced_subset(trainset_full, samples_per_class=250)
    testset = make_balanced_subset(testset_full, samples_per_class=50)

    # Partition dataset
    n_party = args.n_party
    batch_size = args.batch_size
    train_dataset_list, train_loader_list = partition_dataset(
        args.dataset_name, trainset, n_party, batch_size, shuffle=True
    )
    # This is for training. We'll do the forward/backward from these batches:
    pre_extracted_batches = [list(dl) for dl in train_loader_list]
    n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

    # We'll create fresh data loaders for measuring training/test accuracy:
    train_eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_eval_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Build objects
    loss_fn = nn.CrossEntropyLoss()
    ZOO = ZOOutputOptim(mu, args.u_type, args.client_output_size, n_directions=args.sample_times)
    class DummyCompressor:
        def compress_decompress(self, x):
            return x
    compressor = DummyCompressor()
    response_compressor = DummyCompressor()

    # Build models + optimizers
    models = init_models(args.model_type, n_party, train_dataset_list, device, 
                         args.client_output_size, args.server_embedding_size)
    optimizers = init_optimizers(models, server_lr, client_lr)

    print("Starting training...")
    for epoch in range(n_epoch_trial):
        # Train loop
        for batch_idx in range(n_training_batch_per_epoch_client):
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            hidden_feats = {}
            labels_for_loss = None

            # Forward pass for each client (1..n_party-1)
            for m in range(1, n_party):
                inp_m, lab_m = pre_extracted_batches[m][batch_idx]
                inp_m = inp_m.to(device)
                lab_m = lab_m.to(device)
                if m == 1:
                    labels_for_loss = lab_m

                # FO backbone
                # (We do .train() above, so resnet is in training mode)
                feats = models[m].backbone(inp_m)
                # Save for ZO
                hidden_feats[m] = feats.detach()

                # ZO final layer => no FO gradient
                with torch.no_grad():
                    out_full = models[m].client_output_layer(feats)
                out_emb = compress_decompress(out_full, args.compression_type, compressor)
                out_emb_dict[m] = out_emb

            # Combine embeddings at server
            server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
            server_pred = models[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()

            # Update server and client backbones
            for opt in optimizers:
                opt.step()

            # Zero-order update final layer
            for m in range(1, n_party):
                out_original = out_emb_dict[m].detach()
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
                    loss_diff = (loss_plus - loss_minus)
                    loss_diffs.append(loss_diff)

                # est_grad_out is shape [batch, out_dim]
                est_grad_out = ZOO.backward(U, loss_diffs)
                feats = hidden_feats[m].to(device)  # shape [batch, 512] for resnet18
                # dWeight => [out_dim, 512]
                dWeight = torch.matmul(est_grad_out.transpose(0, 1), feats)
                # dBias => [out_dim]
                dBias = est_grad_out.sum(dim=0)

                with torch.no_grad():
                    models[m].client_output_layer.weight -= (client_lr * dWeight)
                    models[m].client_output_layer.bias   -= (client_lr * dBias)

        # ------------------------------
        # End-of-epoch: Evaluate *true* training accuracy
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
                    out_list.append(models[m](inp))  # FO+ZO model
                server_in_eval = torch.cat(out_list, dim=-1)
                out_server = models[0](server_in_eval)
                _, pred = torch.max(out_server, 1)
                train_total += lbl.size(0)
                train_correct += (pred == lbl).sum().item()
        train_acc = 100.0 * train_correct / train_total if train_total else 0

        # Evaluate test accuracy
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

        print(f"Epoch [{epoch}/{n_epoch_trial}]  Train Acc = {train_acc:.2f}%   Test Acc = {test_acc:.2f}%")

        # Switch back to train mode
        models[0].train()
        for m in range(1, n_party):
            models[m].train()

    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# from torch.utils.data import Subset, DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import random
# import argparse
# import time

# ###############################################################################
# # Zero-Order helper
# ###############################################################################
# class ZOOutputOptim:
#     """
#     Zero-order gradient estimator for an output layer.
#     """
#     def __init__(self, mu, u_type, output_dim, n_directions=1):
#         self.mu = mu
#         self.u_type = u_type
#         self.output_dim = output_dim
#         self.n_directions = n_directions

#     def forward_example(self, original_out):
#         """
#         Return lists of (u, out_plus, out_minus) for multiple random directions.
#         """
#         U = []
#         plus_list = []
#         minus_list = []
#         for _ in range(self.n_directions):
#             if self.u_type == "Uniform":
#                 # Some use torch.rand_like(...) or uniform_(-1,1). 
#                 # But Gaussian is often okay for ZO, too:
#                 u = torch.randn_like(original_out)
#             else:
#                 # same as above for illustration
#                 u = torch.randn_like(original_out)

#             out_plus = original_out + self.mu * u
#             out_minus = original_out - self.mu * u
#             U.append(u)
#             plus_list.append(out_plus)
#             minus_list.append(out_minus)
#         return U, plus_list, minus_list

#     def backward(self, U, loss_diffs):
#         """
#         For each direction i:
#            grad_i = (loss_plus_i - loss_minus_i) / (2 * mu) * U[i]
#         Then average across n_directions.
#         """
#         grad_accum = 0.0
#         for i in range(self.n_directions):
#             grad_i = (loss_diffs[i] / (2.0 * self.mu)) * U[i]
#             grad_accum += grad_i
#         grad_est = grad_accum / float(self.n_directions)
#         return grad_est

# ###############################################################################
# # Balanced subset helper for CIFAR-10
# ###############################################################################
# def make_balanced_subset(dataset, samples_per_class):
#     """
#     Return a Subset with samples_per_class examples per class.
#     """
#     class_indices = [[] for _ in range(10)]
#     for idx, (data, label) in enumerate(dataset):
#         if isinstance(label, torch.Tensor):
#             label = label.item()
#         class_indices[label].append(idx)
#     final_indices = []
#     for c in range(10):
#         random.shuffle(class_indices[c])
#         final_indices.extend(class_indices[c][:samples_per_class])
#     return Subset(dataset, final_indices)

# ###############################################################################
# # (Example) compress_decompress utility
# ###############################################################################
# def compress_decompress(tensor, ctype, compressor):
#     if ctype == "None":
#         return tensor
#     return compressor.compress_decompress(tensor)

# ###############################################################################
# # Parse command-line arguments
# ###############################################################################
# parser = argparse.ArgumentParser()
# parser.add_argument('--random_seed', type=int, default=12341)
# parser.add_argument('--framework_type', type=str, default="CZOFO")
# parser.add_argument('--dataset_name', type=str, default="CIFAR10")
# parser.add_argument('--model_type', type=str, default="SimpleResNet18")
# parser.add_argument('--n_party', type=int, default=2)
# parser.add_argument('--client_output_size', type=int, default=10)
# parser.add_argument('--server_embedding_size', type=int, default=128)
# parser.add_argument('--client_lr', type=float, default=0.01)
# parser.add_argument('--server_lr', type=float, default=0.01)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--n_epoch', type=int, default=100)
# parser.add_argument('--u_type', type=str, default="Uniform")
# parser.add_argument('--mu', type=float, default=0.000001)
# parser.add_argument('--d', type=float, default=1)
# parser.add_argument('--sample_times', type=int, default=20)
# parser.add_argument('--compression_type', type=str, default="None")
# parser.add_argument('--compression_bit', type=int, default=8)
# parser.add_argument('--response_compression_type', type=str, default="None")
# parser.add_argument('--response_compression_bit', type=int, default=8)
# parser.add_argument('--local_update_times', type=int, default=1)
# parser.add_argument('--log_file_name', type=str, default="None")
# parser.add_argument('--attack_type', type=str, default=None)
# parser.add_argument('--severity', type=float, default=0.1)
# args = parser.parse_args()

# run_epochs = args.n_epoch
# random.seed(args.random_seed)
# torch.manual_seed(args.random_seed)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ###############################################################################
# # Dataset and Partition Helpers
# ###############################################################################
# def get_dataset(name):
#     """
#     Returns train_dataset and test_dataset with CIFAR-10 transformations.
#     """
#     if name == "CIFAR10":
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2470, 0.2435, 0.2616)),
#         ])
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2470, 0.2435, 0.2616)),
#         ])
#         trainset_full = datasets.CIFAR10(root="./data", train=True,
#                                          download=True, transform=transform_train)
#         testset_full = datasets.CIFAR10(root="./data", train=False,
#                                         download=True, transform=transform_test)
#         return trainset_full, testset_full
#     return None, None

# from torch.utils.data import DataLoader, Dataset

# def partition_dataset(name, dataset, n_party, batch_size, shuffle=True):
#     """
#     Partition 'dataset' into n_party subsets. In standard 'vertical' FL,
#     each party sees the *same examples* but different features. Here,
#     for demonstration, we just replicate 'dataset' for each party.
#     """
#     dataset_list = []
#     loaders = []
#     for i in range(n_party):
#         dataset_list.append(dataset)
#         dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#         loaders.append(dl)
#     return dataset_list, loaders

# ###############################################################################
# # Model and Optimizer Initialization
# ###############################################################################
# def init_models(model_type, n_party, train_dataset_list, device, client_output_size, server_embedding_size):
#     """
#     Build server and client models.
#     models[0]: server model.
#     models[1..n_party-1]: client models with a backbone + client_output_layer.
#     """
#     class ServerModel(nn.Module):
#         def __init__(self, input_dim=client_output_size*(n_party-1), num_classes=10):
#             super().__init__()
#             self.fc = nn.Linear(input_dim, num_classes)
#         def forward(self, x):
#             return self.fc(x)

#     class ClientModel(nn.Module):
#         def __init__(self, backbone_in_dim=32*32*3, hidden_dim=512, out_dim=10):
#             super().__init__()
#             self.backbone = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(backbone_in_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             self.client_output_layer = nn.Linear(hidden_dim, out_dim)

#         def forward(self, x):
#             feats = self.backbone(x)
#             out = self.client_output_layer(feats)
#             return out

#     models = []
#     server_in_dim = (n_party - 1) * client_output_size
#     server_model = ServerModel(input_dim=server_in_dim).to(device)
#     models.append(server_model)
#     for i in range(1, n_party):
#         cm = ClientModel().to(device)
#         models.append(cm)
#     return models

# def init_optimizers(models, server_lr, client_lr):
#     """
#     - Server: FO on all parameters.
#     - Clients: FO on backbone only (exclude final layer).
#     """
#     opts = []
#     # Server optimizer
#     server_params = list(models[0].parameters())
#     server_optimizer = torch.optim.Adam(server_params, lr=server_lr)
#     opts.append(server_optimizer)

#     # Client optimizers
#     for m in range(1, len(models)):
#         backbone_params = []
#         final_layer_params = []
#         for name, param in models[m].named_parameters():
#             if "client_output_layer" in name:
#                 final_layer_params.append(param)
#             else:
#                 backbone_params.append(param)
#         # Zero-order for final layer => disable FO
#         for p in final_layer_params:
#             p.requires_grad = False
#         client_optimizer = torch.optim.Adam(backbone_params, lr=client_lr)
#         opts.append(client_optimizer)
#     return opts

# def append_log(log_item, log_file):
#     pass

# ###############################################################################
# # Main training function
# ###############################################################################
# def main():
#     # Hyperparameters
#     client_lr = args.client_lr
#     server_lr = args.server_lr
#     mu = args.mu
#     n_epoch_trial = run_epochs

#     # Re-seed
#     random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)

#     # Load dataset
#     trainset_full, testset_full = get_dataset(args.dataset_name)
#     # You can do smaller subsets if you want to debug quickly:
#     # trainset = make_balanced_subset(trainset_full, samples_per_class=5000)
#     # testset  = make_balanced_subset(testset_full,  samples_per_class=1000)

#     trainset = make_balanced_subset(trainset_full, samples_per_class=25000)
#     testset = make_balanced_subset(testset_full, samples_per_class=5000)

#     # Partition dataset
#     n_party = args.n_party
#     batch_size = args.batch_size
#     train_dataset_list, train_loader_list = partition_dataset(
#         args.dataset_name, trainset, n_party, batch_size, shuffle=True
#     )
#     # This is for training. We'll do the forward/backward from these batches:
#     pre_extracted_batches = [list(dl) for dl in train_loader_list]
#     n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

#     # We'll create fresh data loaders for measuring training/test accuracy:
#     train_eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
#     test_eval_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#     # Build objects
#     loss_fn = nn.CrossEntropyLoss()
#     ZOO = ZOOutputOptim(mu, args.u_type, args.client_output_size, n_directions=args.sample_times)
#     class DummyCompressor:
#         def compress_decompress(self, x):
#             return x
#     compressor = DummyCompressor()
#     response_compressor = DummyCompressor()

#     # Build models + optimizers
#     models = init_models(args.model_type, n_party, train_dataset_list, device, 
#                          args.client_output_size, args.server_embedding_size)
#     optimizers = init_optimizers(models, server_lr, client_lr)

#     print("Starting training...")
#     for epoch in range(n_epoch_trial):
#         # Train loop
#         for batch_idx in range(n_training_batch_per_epoch_client):
#             for opt in optimizers:
#                 opt.zero_grad()

#             out_emb_dict = {}
#             hidden_feats = {}
#             labels_for_loss = None

#             # Forward pass for each client (1..n_party-1)
#             for m in range(1, n_party):
#                 inp_m, lab_m = pre_extracted_batches[m][batch_idx]
#                 inp_m = inp_m.to(device)
#                 lab_m = lab_m.to(device)
#                 if m == 1:
#                     labels_for_loss = lab_m

#                 # FO backbone
#                 feats = models[m].backbone(inp_m)
#                 # Save for ZO
#                 hidden_feats[m] = feats.detach()

#                 # ZO final layer => no FO gradient
#                 with torch.no_grad():
#                     out_full = models[m].client_output_layer(feats)
#                 out_emb = compress_decompress(out_full, args.compression_type, compressor)
#                 out_emb_dict[m] = out_emb

#             # Combine embeddings at server
#             server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
#             server_pred = models[0](server_in)
#             loss = loss_fn(server_pred, labels_for_loss)
#             loss.backward()

#             # Update server and client backbones
#             for opt in optimizers:
#                 opt.step()

#             # Zero-order update final layer
#             for m in range(1, n_party):
#                 out_original = out_emb_dict[m].detach()
#                 U, plus_list, minus_list = ZOO.forward_example(out_original)
#                 loss_diffs = []
#                 for d_i in range(ZOO.n_directions):
#                     plus_emb_dict = dict(out_emb_dict)
#                     minus_emb_dict = dict(out_emb_dict)
#                     plus_emb_dict[m] = plus_list[d_i]
#                     minus_emb_dict[m] = minus_list[d_i]

#                     server_in_plus = torch.cat([plus_emb_dict[x] for x in range(1, n_party)], dim=-1)
#                     server_in_minus = torch.cat([minus_emb_dict[x] for x in range(1, n_party)], dim=-1)
#                     with torch.no_grad():
#                         server_pred_plus = models[0](server_in_plus)
#                         server_pred_minus = models[0](server_in_minus)
#                         loss_plus = loss_fn(server_pred_plus, labels_for_loss)
#                         loss_minus = loss_fn(server_pred_minus, labels_for_loss)
#                     loss_diff = (loss_plus - loss_minus)
#                     loss_diffs.append(loss_diff)

#                 # est_grad_out is shape [batch, out_dim]
#                 est_grad_out = ZOO.backward(U, loss_diffs)
#                 feats = hidden_feats[m].to(device)  # shape [batch, hidden_dim]
#                 # dWeight => [out_dim, hidden_dim]
#                 dWeight = torch.matmul(est_grad_out.transpose(0, 1), feats)
#                 # dBias => [out_dim]
#                 dBias = est_grad_out.sum(dim=0)

#                 with torch.no_grad():
#                     models[m].client_output_layer.weight -= client_lr * dWeight
#                     models[m].client_output_layer.bias   -= client_lr * dBias

#         # ------------------------------
#         # End-of-epoch: Evaluate *true* training accuracy
#         # ------------------------------
#         models[0].eval()
#         for m in range(1, n_party):
#             models[m].eval()

#         train_correct = 0
#         train_total = 0
#         with torch.no_grad():
#             for inp, lbl in train_eval_loader:
#                 inp, lbl = inp.to(device), lbl.to(device)
#                 # For "vertical" style, each client sees the same input (in principle).
#                 # We'll produce final embeddings from each client, then combine on server.
#                 out_list = []
#                 for m in range(1, n_party):
#                     out_list.append(models[m](inp))  # FO+ZO model
#                 server_in_eval = torch.cat(out_list, dim=-1)
#                 out_server = models[0](server_in_eval)
#                 _, pred = torch.max(out_server, 1)
#                 train_total += lbl.size(0)
#                 train_correct += (pred == lbl).sum().item()
#         train_acc = 100.0 * train_correct / train_total if train_total else 0

#         # Evaluate test accuracy
#         test_correct = 0
#         test_total = 0
#         with torch.no_grad():
#             for inp, lbl in test_eval_loader:
#                 inp, lbl = inp.to(device), lbl.to(device)
#                 out_list = []
#                 for m in range(1, n_party):
#                     out_list.append(models[m](inp))
#                 server_in_eval = torch.cat(out_list, dim=-1)
#                 out_server = models[0](server_in_eval)
#                 _, predicted = torch.max(out_server, 1)
#                 test_total += lbl.size(0)
#                 test_correct += (predicted == lbl).sum().item()
#         test_acc = 100.0 * test_correct / test_total if test_total else 0

#         print(f"Epoch [{epoch}/{n_epoch_trial}]  Train Acc = {train_acc:.2f}%   Test Acc = {test_acc:.2f}%")

#         # Switch back to train mode
#         models[0].train()
#         for m in range(1, n_party):
#             models[m].train()

#     print(f"Final Test Accuracy: {test_acc:.2f}%")

# if __name__ == "__main__":
#     main()