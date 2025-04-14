import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import random
import argparse
import time
import numpy as np
import os

################################################################################
# 1) MaliciousOptimizer & LabelInference Classes (taken from your code2 / label_inference_attack.py)
################################################################################
class MaliciousOptimizer(torch.optim.Optimizer):
    """
    A malicious optimizer that manipulates gradient norms.
    In your original code, it scales gradients by a factor in [min_ratio, max_ratio],
    aiming to degrade or skew training while still producing valid updates.
    """
    def __init__(self, params, lr=1e-3, gamma_lr_scale_up=1.0,
                 min_ratio=1.0, max_ratio=1.0):
        defaults = dict(lr=lr,
                        gamma_lr_scale_up=gamma_lr_scale_up,
                        min_ratio=min_ratio,
                        max_ratio=max_ratio)
        super().__init__(params, defaults)

    def step_malicious(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Very simple malicious approach: scale grad by some random factor
                scale_factor = random.uniform(group['min_ratio'], group['max_ratio'])
                grad = grad * scale_factor

                # Standard gradient descent update
                p.data = p.data - lr * grad

        return loss

    def step(self, closure=None):
        # By default, do the malicious step
        return self.step_malicious(closure)

class LabelInferenceHead(nn.Module):
    """
    Simple MLP Head for label inference, e.g. from embeddings -> predicted label.
    """
    def __init__(self, in_features, out_features=10):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.fc(x)

def extract_auxiliary_datapoints(X_train, y_train, num_aux_per_class=4):
    """
    X_train: shape [N, C, H, W] for image data
    y_train: shape [N], each label in 0..9 for CIFAR-10
    Return: subset of X_train & y_train with num_aux_per_class from each class
    """
    X_out, y_out = [], []
    # Convert to CPU for indexing, if needed
    X_cpu = X_train.cpu()
    y_cpu = y_train.cpu()

    for c in range(10):
        # Indices of class c
        idx_c = (y_cpu == c).nonzero().view(-1)
        # Shuffle
        idx_c = idx_c[torch.randperm(len(idx_c))]
        # Take up to num_aux_per_class
        take_count = min(num_aux_per_class, len(idx_c))
        selected_idx = idx_c[:take_count]
        X_out.append(X_cpu[selected_idx])
        y_out.append(y_cpu[selected_idx])

    X_out = torch.cat(X_out, dim=0)
    y_out = torch.cat(y_out, dim=0)
    return X_out, y_out

def generate_auxiliary_embeddings(model, aux_data, aux_labels, batch_size=256):
    """
    model: e.g. malicious client model (ResNet partial)
    aux_data: shape [N, C, H, W]
    aux_labels: shape [N]
    Return: (embeddings, aux_labels) => embeddings shape [N, out_dim]
    """
    device = next(model.parameters()).device
    model.eval()

    all_embs = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, len(aux_data), batch_size):
            batch_x = aux_data[i:i+batch_size].to(device)
            batch_emb = model.backbone(batch_x)  # skip final layer, or you can do full model
            all_embs.append(batch_emb.cpu())
            all_labels.append(aux_labels[i:i+batch_size].cpu())

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embs, all_labels

def fine_tune_label_inference_head(model, aux_embeddings, aux_labels,
                                   num_epochs=100, batch_size=128, lr=0.01):
    """
    model: an instance of LabelInferenceHead
    aux_embeddings: shape [N, in_features]
    aux_labels: shape [N]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ds = TensorDataset(aux_embeddings, aux_labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model

def extract_embeddings(model, X_data, batch_size=256):
    """
    model: malicious client model (which has .backbone + .client_output_layer)
    X_data: shape [N, C, H, W]
    Return: shape [N, out_dim(backbone)]
    """
    device = next(model.parameters()).device
    model.eval()

    feats_all = []
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch_x = X_data[i:i+batch_size].to(device)
            feats = model.backbone(batch_x)
            feats_all.append(feats.cpu())
    feats_all = torch.cat(feats_all, dim=0)
    return feats_all

def infer_labels_from_embeddings(model, embeddings, batch_size=256):
    """
    model: label_inference_head
    embeddings: shape [N, in_features]
    Return predicted labels
    """
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())
    return torch.cat(predictions, dim=0)

def compute_label_inference_accuracy(predicted_labels, true_labels):
    correct = (predicted_labels.cpu() == true_labels.cpu()).sum().item()
    total = true_labels.size(0)
    return correct / total

def compute_topk_accuracy(outputs, targets, k=5):
    """
    outputs: shape [N, num_classes], raw logits
    targets: shape [N], ground-truth labels
    Return fraction of samples whose label is in top-k predictions
    """
    device = outputs.device
    targets = targets.to(device)
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    topk_correct = correct.any(dim=1).float().sum()
    acc = topk_correct.item() / targets.size(0)
    return acc

def infer_labels_from_embeddings_topk(model, embeddings, batch_size=256):
    """
    Return raw logits from the label_inference_head for top-k computations.
    """
    device = next(model.parameters()).device
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            outputs = model(batch)
            all_outputs.append(outputs.cpu())
    return torch.cat(all_outputs, dim=0)

################################################################################
# 2) Zero-Order Output Optim for final layer
################################################################################
class ZOOutputOptim:
    """
    Zero-order gradient estimator for the final (client_output) layer.
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
                # For simplicity, treat the same as "Uniform"
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

################################################################################
# 3) Utility function to subset CIFAR-10 (balanced subset)
################################################################################
def make_balanced_subset(dataset, samples_per_class=2500, num_classes=10):
    class_indices = [[] for _ in range(num_classes)]
    for idx, (data, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    final_indices = []
    for c in range(num_classes):
        random.shuffle(class_indices[c])
        final_indices.extend(class_indices[c][:samples_per_class])
    return Subset(dataset, final_indices)

################################################################################
# 4) Vertical split function
################################################################################
def vertical_split_with_adversary(images, total_parties, adversary_idx, adversary_share=0.5):
    if hasattr(images, 'cpu'):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    N, C, H, W = images_np.shape
    adv_cols = int(round(W * adversary_share))

    remaining = W - adv_cols
    normal_clients = total_parties - 1
    if normal_clients <= 0:
        raise ValueError("Need at least 2 parties (server + client).")

    base_normal = remaining // normal_clients
    extra = remaining % normal_clients

    widths = []
    for i in range(total_parties):
        if i == adversary_idx:
            widths.append(adv_cols)
        else:
            w = base_normal + (1 if extra > 0 else 0)
            if extra > 0:
                extra -= 1
            widths.append(w)

    assert sum(widths) == W, "Widths do not sum to original image width!"

    image_parts_np = [
        np.zeros((N, C, H, widths[i]), dtype=np.float32)
        for i in range(total_parties)
    ]

    for n in range(N):
        current_col = 0
        for i in range(total_parties):
            end_col = current_col + widths[i]
            image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
            current_col = end_col

    parts = [torch.from_numpy(ip).float() for ip in image_parts_np]
    return parts

################################################################################
# 5) Argument parser
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12341)
parser.add_argument('--n_party', type=int, default=6)
parser.add_argument('--client_lr', type=float, default=0.001)
parser.add_argument('--server_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--u_type', type=str, default="Uniform")
parser.add_argument('--mu', type=float, default=0.001)
parser.add_argument('--sample_times', type=int, default=5)
parser.add_argument('--attack_type', type=str, default='label_inference',
                    help="Set to 'label_inference' to enable malicious client + saving data.")
parser.add_argument('--malicious_client_idx', type=int, default=1)
parser.add_argument('--adversary_share', type=float, default=0.5)
parser.add_argument('--client_output_size', type=int, default=10)
parser.add_argument('--g_tm', type=float, default=0.525)
parser.add_argument('--g_bm', type=float, default=0.525)
args = parser.parse_args()

################################################################################
# 6) Model definitions
################################################################################
def init_models(n_party, client_output_size):
    """
    First model (index 0) is the server: a simple linear layer.
    Then each client has a ResNet backbone and a final linear layer.
    """
    class ServerModel(nn.Module):
        def __init__(self, input_dim, num_classes=10):
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

    # The server sees concatenation of (n_party-1) client outputs
    input_dim_server = (n_party - 1) * client_output_size
    server_model = ServerModel(input_dim_server).to(device)
    models = [server_model]
    for i in range(1, n_party):
        cm = ClientResNetModel(out_dim=client_output_size, pretrained=False).to(device)
        models.append(cm)
    return models

def init_optimizers(models, n_party, server_lr, client_lr, malicious_client_idx=None, attack_type=None):
    server_optimizer = torch.optim.Adam(models[0].parameters(), lr=server_lr)
    opts = [server_optimizer]
    for i in range(1, n_party):
        if attack_type == 'label_inference' and i == malicious_client_idx:
            print(f"[INFO] Client {i} uses MaliciousOptimizer ...")
            opt = MaliciousOptimizer(models[i].parameters(), lr=client_lr,
                                     gamma_lr_scale_up=0.7, min_ratio=1, max_ratio=5)
        else:
            opt = torch.optim.Adam(models[i].parameters(), lr=client_lr)
        opts.append(opt)
    return opts

################################################################################
# main()
################################################################################
def main():
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_party = args.n_party
    if n_party < 2:
        raise ValueError("Need at least 2 parties (server + at least 1 client).")

    # 1) Download CIFAR-10 and pick balanced subsets for train + test
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
    trainset_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset_full  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    trainset = make_balanced_subset(trainset_full, samples_per_class=2500)  # ~ 25000
    testset  = make_balanced_subset(testset_full,  samples_per_class=1000)  # ~ 10000

    # Collect all train images, labels => single big batch
    train_loader_full = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_loader_full  = DataLoader(testset,  batch_size=len(testset),  shuffle=False)

    train_images, train_labels = next(iter(train_loader_full))
    test_images,  test_labels  = next(iter(test_loader_full))
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    test_images,  test_labels  = test_images.to(device), test_labels.to(device)

    # 2) Vertical split
    splitted_train = vertical_split_with_adversary(train_images,
                                                   total_parties=n_party,
                                                   adversary_idx=args.malicious_client_idx,
                                                   adversary_share=args.adversary_share)
    splitted_test  = vertical_split_with_adversary(test_images,
                                                   total_parties=n_party,
                                                   adversary_idx=args.malicious_client_idx,
                                                   adversary_share=args.adversary_share)

    # 3) Create DataLoaders (for each client). We'll store all in memory as well
    client_loaders_train = []
    for i in range(n_party):
        ds = TensorDataset(splitted_train[i].cpu(), train_labels.cpu())
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        client_loaders_train.append(dl)

    client_loaders_test = []
    for i in range(n_party):
        ds_test = TensorDataset(splitted_test[i].cpu(), test_labels.cpu())
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
        client_loaders_test.append(dl_test)

    # turn them into lists of (images,labels) for easy indexing
    pre_extracted_batches_train = [list(dl) for dl in client_loaders_train]
    pre_extracted_batches_test  = [list(dl) for dl in client_loaders_test]

    # 4) Init models + optimizers
    models = init_models(n_party, args.client_output_size)
    optimizers = init_optimizers(models, n_party,
                                 server_lr=args.server_lr,
                                 client_lr=args.client_lr,
                                 malicious_client_idx=args.malicious_client_idx,
                                 attack_type=args.attack_type)

    # 5) Zero-Order helper for final layer
    ZOO = ZOOutputOptim(mu=args.mu,
                        u_type=args.u_type,
                        output_dim=args.client_output_size,
                        n_directions=args.sample_times)

    loss_fn = nn.CrossEntropyLoss()
    n_epoch = args.n_epoch

    # Training
    print(f"\n[INFO] Start FO+ZO training with attack_type={args.attack_type}, n_party={n_party}\n")
    n_training_batches = len(pre_extracted_batches_train[1])  # assume party1 dictates batch count

    for epoch in range(n_epoch):
        # set models to train
        for m in range(n_party):
            models[m].train()

        for batch_idx in range(n_training_batches):
            # zero_grad all
            for opt in optimizers:
                opt.zero_grad()

            out_emb_dict = {}
            hidden_feats = {}
            labels_for_loss = None

            # forward for each client i in [1..n_party-1]
            for i in range(1, n_party):
                (imgs, labs) = pre_extracted_batches_train[i][batch_idx]
                imgs = imgs.to(device)
                labs = labs.to(device)
                if i == 1:
                    labels_for_loss = labs
                # backbone forward
                feats = models[i].backbone(imgs)
                hidden_feats[i] = feats.detach()
                # FO final-layer forward
                out_full = models[i].client_output_layer(feats)
                out_emb_dict[i] = out_full

            # server forward
            all_outs = [out_emb_dict[i] for i in range(1, n_party)]
            server_in = torch.cat(all_outs, dim=-1)
            server_pred = models[0](server_in)
            loss = loss_fn(server_pred, labels_for_loss)
            loss.backward()

            # step FO optimizers
            for opt in optimizers:
                opt.step()

            # ZO for final layer
            for i in range(1, n_party):
                with torch.no_grad():
                    out_original = models[i].client_output_layer(hidden_feats[i])
                U, plus_list, minus_list = ZOO.forward_example(out_original)
                loss_diffs = []
                for d_i in range(ZOO.n_directions):
                    plus_emb = dict(out_emb_dict)
                    minus_emb = dict(out_emb_dict)
                    plus_emb[i]  = plus_list[d_i]
                    minus_emb[i] = minus_list[d_i]

                    sp_in_plus  = torch.cat([plus_emb[x] for x in range(1, n_party)], dim=-1)
                    sp_in_minus = torch.cat([minus_emb[x] for x in range(1, n_party)], dim=-1)
                    with torch.no_grad():
                        sp_plus  = models[0](sp_in_plus)
                        sp_minus = models[0](sp_in_minus)
                        l_plus  = loss_fn(sp_plus, labels_for_loss)
                        l_minus = loss_fn(sp_minus, labels_for_loss)
                    loss_diffs.append(l_plus - l_minus)

                est_grad_out = ZOO.backward(U, loss_diffs)
                feats = hidden_feats[i]
                dWeight = torch.matmul(est_grad_out.transpose(0,1), feats.to(device))
                dBias   = est_grad_out.sum(dim=0)
                lr = args.client_lr
                with torch.no_grad():
                    models[i].client_output_layer.weight -= lr * dWeight
                    models[i].client_output_layer.bias   -= lr * dBias

        # Evaluate at epoch end
        for m in range(n_party):
            models[m].eval()

        # Train accuracy
        train_correct, train_total = 0, 0
        for bidx in range(len(pre_extracted_batches_train[1])):
            (dummy_imgs, labs) = pre_extracted_batches_train[1][bidx]
            labs = labs.to(device)
            tmp_outs = []
            for i in range(1, n_party):
                (im_i, _) = pre_extracted_batches_train[i][bidx]
                out_i = models[i](im_i.to(device))
                tmp_outs.append(out_i)
            sv_in = torch.cat(tmp_outs, dim=-1)
            sv_pred = models[0](sv_in)
            _, pred_label = torch.max(sv_pred, 1)
            train_total += labs.size(0)
            train_correct += (pred_label == labs).sum().item()

        train_acc = 100.0 * train_correct / train_total if train_total>0 else 0

        # Test accuracy
        test_correct, test_total = 0, 0
        for bidx in range(len(pre_extracted_batches_test[1])):
            (dummy_imgs, labs) = pre_extracted_batches_test[1][bidx]
            labs = labs.to(device)
            tmp_outs = []
            for i in range(1, n_party):
                (im_i, _) = pre_extracted_batches_test[i][bidx]
                out_i = models[i](im_i.to(device))
                tmp_outs.append(out_i)
            sv_in = torch.cat(tmp_outs, dim=-1)
            sv_pred = models[0](sv_in)
            _, pred_label = torch.max(sv_pred, 1)
            test_total += labs.size(0)
            test_correct += (pred_label == labs).sum().item()

        test_acc = 100.0 * test_correct / test_total if test_total>0 else 0

        print(f"Epoch [{epoch+1}/{n_epoch}] Train Acc = {train_acc:.2f}% | Test Acc = {test_acc:.2f}%")

    print(f"[INFO] Final Test Accuracy = {test_acc:.2f}%")

    ############################################################################
    # 7) If label_inference attack -> save data + malicious model + PRINT ACCURACY
    ############################################################################
    if args.attack_type == 'label_inference':
        print("[INFO] Saving data for label inference + running label inference evaluation ...")
        os.makedirs("label_inference_data", exist_ok=True)

        # The malicious client’s columns
        X_train_mal = splitted_train[args.malicious_client_idx].cpu()
        X_val_mal   = splitted_test[args.malicious_client_idx].cpu()

        # For demonstration, we store the real training labels
        y_train_cpu = train_labels.cpu()

        # Save for external script if you like
        torch.save(X_train_mal, "label_inference_data/X_train.pt")
        torch.save(X_val_mal,   "label_inference_data/X_val.pt")
        torch.save(y_train_cpu, "label_inference_data/y_train.pt")

        # malicious client's model
        torch.save(models[args.malicious_client_idx].state_dict(),
                   "label_inference_data/model_state_dict.pt")

        # ----------------------
        # 7a) We also run label inference *here* (inline) to print accuracy
        # ----------------------
        malicious_model = models[args.malicious_client_idx]  # ResNet-based
        malicious_model.eval()

        # i) Extract a small "auxiliary" set => e.g. 50 per class
        aux_data, aux_labels = extract_auxiliary_datapoints(
            X_train_mal, y_train_cpu, num_aux_per_class=50
        )

        # ii) Generate embeddings from malicious model's backbone
        aux_embeddings, aux_labels = generate_auxiliary_embeddings(malicious_model,
                                                                   aux_data,
                                                                   aux_labels,
                                                                   batch_size=256)

        # iii) Fine-tune a small label inference head
        in_features = aux_embeddings.shape[1]
        lia_head = LabelInferenceHead(in_features, out_features=10).to(device)
        lia_head = fine_tune_label_inference_head(
            lia_head, aux_embeddings, aux_labels,
            num_epochs=50, batch_size=128, lr=0.01
        )

        # iv) Extract embeddings for the "full training set" portion for malicious client
        #     (or you could do the val set, etc.)
        full_train_embeddings = extract_embeddings(malicious_model, X_train_mal, batch_size=256)

        # v) Predict labels
        pred_labels = infer_labels_from_embeddings(lia_head, full_train_embeddings, batch_size=256)
        top1_acc = compute_label_inference_accuracy(pred_labels, y_train_cpu)

        # For top-k accuracy, we need raw logits
        full_train_logits = infer_labels_from_embeddings_topk(lia_head, full_train_embeddings, batch_size=256)
        top5_acc = compute_topk_accuracy(full_train_logits, y_train_cpu, k=5)

        print(f"[Label Inference]   top-1 accuracy = {100*top1_acc:.2f}%")
        print(f"[Label Inference]   top-5 accuracy = {100*top5_acc:.2f}%")

if __name__ == "__main__":
    main()