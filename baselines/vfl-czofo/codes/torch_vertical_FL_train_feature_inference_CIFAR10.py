import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models  # for ResNet
import random
import argparse
import time

def feature_inference_attack(victim_model, dataset, device, num_images=4, steps=500, lr=0.01):
    """
    Perform a naive embedding-inversion attack on 'victim_model' for a small batch of data.
    
    :param victim_model: nn.Module (the client model we want to invert)
    :param dataset: the dataset from which we pick images to invert
    :param device: 'cuda' or 'cpu'
    :param num_images: how many images to invert at once
    :param steps: gradient descent steps
    :param lr: learning rate for the attack
    """
    victim_model.eval()
    
    # 1) Pick a small batch from the victim's dataset
    indices = random.sample(range(len(dataset)), num_images)
    real_imgs = []
    for idx in indices:
        img, _ = dataset[idx]  # ignoring label here
        real_imgs.append(img)
    real_imgs = torch.stack(real_imgs).to(device)  # shape: [num_images, C, H, W]
    
    # 2) Compute the "target embedding" we want to invert
    with torch.no_grad():
        target_embedding = victim_model.backbone(real_imgs)     # shape: [num_images, 512]
        target_embedding = victim_model.client_output_layer(target_embedding)  # shape: [num_images, out_dim]
    
    # 3) Create a random input variable to optimize
    x_recon = torch.randn_like(real_imgs, requires_grad=True)  # same shape as real_imgs
    
    # 4) Gradient-based attack: optimize x_recon so that the victim_model(x_recon) matches target_embedding
    optimizer = torch.optim.Adam([x_recon], lr=lr)
    loss_fn = nn.MSELoss()
    
    for step in range(steps):
        optimizer.zero_grad()
        pred_embedding = victim_model.backbone(x_recon)
        pred_embedding = victim_model.client_output_layer(pred_embedding)
        
        # 4A) MSE between predicted embedding and target embedding
        loss = loss_fn(pred_embedding, target_embedding)
        loss.backward()
        
        optimizer.step()
        
        # Optionally clamp or regularize x_recon if you don’t want values to blow up
        x_recon.data = torch.clamp(x_recon.data, -3.0, 3.0)
        
        if (step+1) % 50 == 0:
            print(f" Attack step {step+1}/{steps}, MSE embedding loss = {loss.item():.4f}")
    
    # 5) Return the final reconstructed images and the real images
    return x_recon.detach(), real_imgs.detach()


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
# Parse-command-line arguments
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12341)
parser.add_argument('--framework_type', type=str, default="CZOFO")
parser.add_argument('--dataset_name', type=str, default="CIFAR10")
parser.add_argument('--model_type', type=str, default="SimpleResNet18")
parser.add_argument('--n_party', type=int, default=4)
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
            backbone.fc = nn.Identity()  # remove the original FC layer
            self.backbone = backbone
            self.client_output_layer = nn.Linear(512, out_dim)
        def forward(self, x):
            feats = self.backbone(x)
            out = self.client_output_layer(feats)
            return out

    models = []
    server_in_dim = (n_party - 1) * client_output_size
    server_model = ServerModel(input_dim=server_in_dim).to(device)
    models.append(server_model)
    for i in range(1, n_party):
        cm = ClientResNetModel(out_dim=client_output_size, pretrained=False).to(device)
        models.append(cm)
    return models

def init_optimizers(models, server_lr, client_lr):
    # Now, we use pure FO optimizers for all parameters (including final layer)
    opts = []
    server_optimizer = torch.optim.Adam(models[0].parameters(), lr=server_lr)
    opts.append(server_optimizer)
    for m in range(1, len(models)):
        client_optimizer = torch.optim.Adam(models[m].parameters(), lr=client_lr)
        opts.append(client_optimizer)
    return opts

def append_log(log_item, log_file):
    pass

###############################################################################
# Main training function with FO + ZO reintroduced
###############################################################################
def main():
    client_lr = args.client_lr
    server_lr = args.server_lr
    mu = args.mu
    n_epoch_trial = run_epochs

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    trainset_full, testset_full = get_dataset(args.dataset_name)
    # For debugging overfitting, use a small subset:
    trainset = make_balanced_subset(trainset_full, samples_per_class=2500)  # 50*10=500 samples
    testset = make_balanced_subset(testset_full, samples_per_class=1000)     # 10*10=100 samples

    n_party = args.n_party
    batch_size = args.batch_size
    train_dataset_list, train_loader_list = partition_dataset(args.dataset_name, trainset, n_party, batch_size, shuffle=True)
    pre_extracted_batches = [list(dl) for dl in train_loader_list]
    n_training_batch_per_epoch_client = len(pre_extracted_batches[0])

    train_eval_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_eval_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    # Create the ZO helper (we'll reintroduce the ZO update for the final layer)
    ZOO = ZOOutputOptim(mu, args.u_type, args.client_output_size, n_directions=args.sample_times)
    class DummyCompressor:
        def compress_decompress(self, x):
            return x
    compressor = DummyCompressor()
    response_compressor = DummyCompressor()

    models = init_models(args.model_type, n_party, train_dataset_list, device, args.client_output_size, args.server_embedding_size)
    optimizers = init_optimizers(models, server_lr, client_lr)

    print("Starting training with FO + ZO updates for the final layer...")
    for epoch in range(n_epoch_trial):
        for batch_idx in range(n_training_batch_per_epoch_client):
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            hidden_feats = {}
            labels_for_loss = None

            # Forward pass: for each client, we compute the backbone output with gradients,
            # then for the final layer, we will compute its output in two ways:
            # (1) For the FO update, we temporarily compute the output with gradients.
            # (2) For the ZO update, we compute a separate forward pass without gradients.
            for m in range(1, n_party):
                inp_m, lab_m = pre_extracted_batches[m][batch_idx]
                inp_m = inp_m.to(device)
                lab_m = lab_m.to(device)
                if m == 1:
                    labels_for_loss = lab_m

                # Compute backbone output with gradients:
                feats = models[m].backbone(inp_m)
                hidden_feats[m] = feats.detach()  # save for ZO update later

                # (1) FO: compute full forward pass (backbone + final layer) normally:
                out_full = models[m].client_output_layer(feats)  # FO computation
                out_emb_dict[m] = out_full

            # Combine outputs at the server:
            server_in = torch.cat([out_emb_dict[m] for m in range(1, n_party)], dim=-1)
            server_pred = models[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()

            

            # Step the FO optimizers (for both server and all client parameters)
            for opt in optimizers:
                opt.step()

            # Reintroduce the ZO update for the final layer:
            for m in range(1, n_party):
                # Print weight norm before ZO update for debugging:
                weight_norm_before = models[m].client_output_layer.weight.norm().item()
                # Compute the output of the final layer (without gradients)
                with torch.no_grad():
                    out_original = models[m].client_output_layer(hidden_feats[m])
                # Compute finite-difference directions:
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
                # Estimate gradient for the final layer's output:
                est_grad_out = ZOO.backward(U, loss_diffs)
                feats = hidden_feats[m].to(device)  # shape: [batch, 512]
                # Compute gradients for final layer parameters:
                dWeight = torch.matmul(est_grad_out.transpose(0, 1), feats)
                dBias = est_grad_out.sum(dim=0)
                with torch.no_grad():
                    models[m].client_output_layer.weight -= client_lr * dWeight
                    models[m].client_output_layer.bias   -= client_lr * dBias
                weight_norm_after = models[m].client_output_layer.weight.norm().item()
                # print(f"Epoch {epoch}, Client {m} final layer weight norm: before = {weight_norm_before:.4f}, after = {weight_norm_after:.4f}")

        # Evaluation at end of epoch:
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

        print(f"Epoch [{epoch}/{n_epoch_trial}]  FO+ZO Train Acc = {train_acc:.2f}%   Test Acc = {test_acc:.2f}%")

        models[0].train()
        for m in range(1, n_party):
            models[m].train()

    print(f"Final Test Accuracy: {test_acc:.2f}%")

    victim_id = 1  # Suppose we invert the first client
    victim_dataset = testset  # or trainset, whichever you want to test
    victim_model = models[victim_id]  # from your list: models[1], etc.

    print(f"\n[INFO] Running feature-inference attack on client {victim_id} ...")
    x_recon, x_real = feature_inference_attack(
        victim_model=victim_model,
        dataset=victim_dataset,
        device=device,
        num_images=4,
        steps=300,    # tune as needed
        lr=0.01
    )

    # Optional: Evaluate pixel-level MSE or visualize
    mse_pixels = ((x_recon - x_real)**2).mean().item()
    print(f"[ATTACK] Pixel-level MSE: {mse_pixels:.4f}")

if __name__ == "__main__":
    main()