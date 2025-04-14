# -*- coding: utf-8 -*-

import argparse
import time
import pandas as pd
import torch
import numpy as np
import sys
sys.path.append('../../torch_utils/')
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader  # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10
import random
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from backdoor_attack import find_best_patch_location, insert_trigger_patch


def flip_labels(y_train, num_classes, flip_ratio=0.25):
    y_train_flipped = y_train.clone()  # Copy original labels
    num_samples = len(y_train)
    num_flips = int(flip_ratio * num_samples)  # 25% of total samples

    # Randomly select indices to flip
    flip_indices = torch.randperm(num_samples)[:num_flips]

    # Generate random new labels (ensuring they are different from original)
    for idx in flip_indices:
        original_label = y_train_flipped[idx].squeeze().item()  # Squeeze to remove extra dimensions
        new_label = np.random.choice([i for i in range(num_classes) if i != original_label])
        y_train_flipped[idx] = new_label  # Assign new label

    return y_train_flipped

malicious_client_idx = 1
source_class = 3
target_class = 8
trigger_value = 0.02
window_size = 5
default_organization_num = 2


def privacy_transform_grad(param, tau, theta_u, noise_std):
    """
    Modifies param.grad in place using vectorized operations.
    - Adds Gaussian noise to all gradient values.
    - Creates a mask for those values whose absolute value is above tau.
    - If more than theta_u fraction of gradient values meet the condition, randomly selects exactly that many.
    - Otherwise, keeps all values above the threshold.
    """
    if param.grad is None:
        return

    grad = param.grad
    flat_grad = grad.view(-1)
    total = flat_grad.numel()
    desired_count = int(np.ceil(theta_u * total))

    # Vectorized noise addition to all elements
    noise = torch.normal(mean=0, std=noise_std, size=(total,), device=grad.device)
    noisy_vals = flat_grad + noise

    # Create a mask: True if the absolute noisy value exceeds tau.
    mask = noisy_vals.abs() > tau
    valid_count = mask.sum().item()

    # Prepare a new gradient tensor
    new_flat_grad = torch.zeros_like(flat_grad)

    if valid_count >= desired_count:
        # Randomly select desired_count indices among those that satisfy the threshold.
        valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        perm = torch.randperm(valid_indices.numel(), device=grad.device)
        selected = valid_indices[perm[:desired_count]]
        new_flat_grad[selected] = noisy_vals[selected]
    else:
        # If fewer than desired_count satisfy the condition, keep them all.
        new_flat_grad[mask] = noisy_vals[mask]

    # Reshape back to the original gradient shape.
    param.grad.data = new_flat_grad.view_as(grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
default_organization_num = 2
one_hot = False

class Vertical_FL_Train:
    ''' Vertical Federated Learning Training Class'''
    def __init__(self, active_clients):
        # We will not use active_clients in DP-SGD integration.
        if active_clients is None:
            self.active_clients = [True] * (default_organization_num - 1)
        else:
            self.active_clients = active_clients
        
    def run(self, args, learning_rates, batch_size):
        ''' Main function for the program'''
        data_type = args.data_type                  
        model_type = args.model_type                
        epochs = args.epochs 
        poisoning_budget = args.poisoning_budget

        print("num_default_epochs: ", epochs)
        
        organization_num = args.organization_num    
        attribute_split_array = np.zeros(organization_num).astype(int)  
        
        # Dataset preprocessing (using CIFAR-10)
        if data_type == 'original':
            if args.dname == 'CIFAR10':
                transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])
                    ])
                transfor_val = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])
                    ])
                train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
                img, label = train_set[0]
                test_set = CIFAR10(root='./data', train=False, download=True, transform=transfor_val)
                
                train_loader = DataLoader(train_set, len(train_set))
                test_loader = DataLoader(test_set, len(test_set))
                
                train_images, train_labels = next(iter(train_loader))
                test_images, test_labels = next(iter(test_loader))
                X = torch.cat((train_images, test_images), dim=0)
                y = torch.cat((train_labels, test_labels), dim=0)
                print("y[:10]", y[:10])
                
                samples_per_class = 3000
                y_np = y.numpy()
                balanced_indices = []
                num_classes = 10
                for cls in range(num_classes):
                    cls_indices = np.where(y_np == cls)[0]
                    np.random.shuffle(cls_indices)
                    selected = cls_indices[:samples_per_class]
                    balanced_indices.extend(selected.tolist())
                balanced_indices = np.array(balanced_indices)
                print("Total balanced subset size:", len(balanced_indices))
                X_balanced = X[balanced_indices]
                y_balanced = y[balanced_indices]
                print("Balanced X shape:", X_balanced.shape)
                print("Balanced y shape:", y_balanced.shape)
        else:
            file_path = "./dataset/{0}.dat".format(args.dname)
            X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
        
        # Initialize result arrays
        train_loss_array = []
        val_loss_array = []
        val_auc_array = []
        train_auc_array = []
        X = X_balanced
        y = y_balanced

        # Vertical splitting: split the width of the images among organizations.
        images_np = X.cpu().numpy()  # shape: (N, 3, 32, 32)
        N = images_np.shape[0]
        base_width = 32 // organization_num
        remainder = 32 % organization_num
        widths = [base_width + (1 if i < remainder else 0) for i in range(organization_num)]
        image_parts_np = [np.zeros((N, 3, 32, widths[i]), dtype=np.float32) for i in range(organization_num)]
        for n in range(N):
            current_col = 0
            for i in range(organization_num):
                end_col = current_col + widths[i]
                image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
                current_col = end_col
        encoded_vertical_splitted_data = image_parts_np
        print('X shape:', X.shape)
        random_seed = 1001
        
        # Split the data for each organization
        X_train_vertical_FL = {}
        X_val_vertical_FL = {}
        for organization_idx in range(organization_num):
            if organization_idx == 0:
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                y_train_val = y
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
                    train_test_split(X_train_val, y_train_val, test_size=10000/60000, random_state=random_seed)
            else:
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                dummy_labels = np.zeros(len(X_train_val))
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
                    train_test_split(X_train_val, dummy_labels, test_size=10000/60000, random_state=random_seed)
        
        train_loader_list, test_loader_list, val_loader_list = [], [], []
        for organization_idx in range(organization_num):
            X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
            X_val_vertical_FL[organization_idx] = torch.from_numpy(X_val_vertical_FL[organization_idx]).float()
            print("X_train_vertical_FL[organization_idx]: ", X_train_vertical_FL[organization_idx].shape)
            print("X_val_vertical_FL[organization_idx]: ", X_val_vertical_FL[organization_idx].shape)
            train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size, shuffle=True))
            val_loader_list.append(DataLoader(X_val_vertical_FL[organization_idx], batch_size=batch_size, shuffle=False))
        y_train = torch.from_numpy(y_train.numpy()).long()
        y_val = torch.from_numpy(y_val.numpy()).long()
        train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
        val_loader_list.append(DataLoader(y_val, batch_size=batch_size))
        y_train = y_train.to(device)
        y_val = y_val.to(device)
        # NN architecture setup
        num_organization_hidden_units = 128
        organization_hidden_units_array = [np.array([num_organization_hidden_units])]*organization_num
        organization_output_dim = np.array([64 for i in range(organization_num)])
        top_hidden_units = np.array([512,256,64])
        top_output_dim = 10
        
        # Build organization (client) models
        organization_models = {}
        for organization_idx in range(organization_num):
            organization_models[organization_idx] = torch_organization_model_cifar10(
                out_dim=organization_output_dim[organization_idx]
            ).to(device)
        print("organization_output_dim: ", organization_output_dim)
        
        # Build the top model. Here we assume that *all* organizations are active and we concatenate their outputs.
        top_model_input_dim = 64 * organization_num
        top_model = torch_top_model_cifar10(top_model_input_dim, top_hidden_units, top_output_dim)
        top_model = top_model.to(device).float()

        optimizer = torch.optim.SGD(
            top_model.parameters(),
            lr=learning_rates[0],
            momentum=0.9,
            weight_decay=5e-4
        )

        # Update bottom (organization) model optimizers
        optimizer_organization_list = []
        for organization_idx in range(organization_num):
            lr_for_client = learning_rates[1]
            optimizer_organization_list.append(
                torch.optim.SGD(
                    organization_models[organization_idx].parameters(),
                    lr=lr_for_client,
                    momentum=0.9,
                    weight_decay=1e-3
                )
            )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[15, 30, 40, 60, 80], 
            gamma=args.g_tm
        )

        scheduler_organization_list = [
            torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 30, 40, 60, 80], gamma=args.g_bm)
            for opt in optimizer_organization_list
        ]


        saliency_maps = np.load('saliency_maps.npy')
        print("Loaded saliency maps shape:", saliency_maps.shape)

        source_indices = (y_train == source_class).nonzero(as_tuple=True)[0]
        target_indices = (y_train == target_class).nonzero(as_tuple=True)[0]

        poison_source_indices = source_indices
        num_target_to_poison = int(poisoning_budget * len(saliency_maps))
        poison_target_indices = target_indices[:num_target_to_poison]
        
        trigger_val = 0.01
        X_second_half = X_train_vertical_FL[1]

        source_indices = (y_train == source_class).nonzero(as_tuple=True)[0]
        num_poison = int(poisoning_budget * len(source_indices))
        selected_poison_indices = source_indices[torch.randperm(len(source_indices))[:num_poison]]

        for idx in selected_poison_indices:
            sal_map = saliency_maps[idx]
            img_tensor = X_second_half[idx]
            r0, c0 = find_best_patch_location(sal_map, window_size)
            triggered_img = insert_trigger_patch(img_tensor.clone(), r0, c0, window_size, trigger_val)
            X_second_half[idx] = triggered_img


        print(f"Selected {len(selected_poison_indices)} images for backdoor insertion.")
        print("First 10 indices:", selected_poison_indices[:10])

        y_train_poisoned = y_train.clone()
        y_train_poisoned[selected_poison_indices] = target_class

        y_train = y_train_poisoned
        X_train_vertical_FL[1] = X_second_half 
        
        print('\nStart vertical FL......\n')   
        criterion = nn.CrossEntropyLoss()
        top_model.train()

        
        # Training loop with DP-SGD integration
        asr_array = []
        for i in range(epochs):
            print('Epoch: ', i)
            print("Current top model LR:", optimizer.param_groups[0]['lr'])
            st = time.time()
            batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
            total_asr = 0
            total_misclassified = 0
            for bidx in range(len(batch_idxs_list)):
                batch_idxs = batch_idxs_list[bidx]
                train_auc_array_temp = []
                # Zero out gradients for top and bottom models
                optimizer.zero_grad()
                for organization_idx in range(organization_num):
                    optimizer_organization_list[organization_idx].zero_grad()

                organization_outputs = {}
                for organization_idx in range(organization_num):
                    data_batch = X_train_vertical_FL[organization_idx][batch_idxs].to(device)
                    organization_outputs[organization_idx] = organization_models[organization_idx](data_batch)
                    
                # Concatenate outputs from all organizations
                organization_outputs_cat = torch.cat([organization_outputs[j] for j in range(organization_num)], dim=1).float()
                organization_outputs_cat = organization_outputs_cat.float()
                
                outputs = top_model(organization_outputs_cat)
                y_train = y_train.to(device)
                log_probs = outputs
                train_loss = criterion(log_probs, y_train[batch_idxs])
                
                predictions = torch.argmax(log_probs, dim=1)
                correct = (predictions == y_train[batch_idxs]).sum().item()
                total = y_train[batch_idxs].size(0)
                train_auc = correct / total
                train_auc_array_temp.append(train_auc)
                
                # ---------- Privacy-Preserving Gradient Transformation Start ----------
                # Perform backward pass as usual
                # train_loss.backward()

                labels_batch = y_train[batch_idxs].to(device)
                second_feats = organization_outputs[1]  # second party embeddings

                source_mask = (labels_batch == source_class)
                target_mask = (labels_batch == target_class)



                if source_mask.sum() > 0 and target_mask.sum() > 0:
                    source_mean = second_feats[source_mask].mean(dim=0)
                    target_mean = second_feats[target_mask].mean(dim=0)
                    l2_distance = torch.norm(source_mean - target_mean, p=2)
                else:
                    l2_distance = torch.tensor(0.0, device=device)

                # Regularization weight (tune as needed)
                lambda_reg = 0.1

                if i < 30:
                    combined_loss = train_loss
                else:
                    combined_loss = train_loss + lambda_reg * l2_distance


                batch_misclassified = (predictions == target_class).sum().item()
                total_misclassified += batch_misclassified
                
                combined_loss.backward()

                # For the top model:
                for param in top_model.parameters():
                    privacy_transform_grad(param, tau=args.tau, theta_u=args.theta_u, noise_std=args.noise_std)

                # For each organization model:
                for organization_idx in range(organization_num):
                    for param in organization_models[organization_idx].parameters():
                        privacy_transform_grad(param, tau=args.tau, theta_u=args.theta_u, noise_std=args.noise_std)

                optimizer.step()
                for organization_idx in range(organization_num):
                    optimizer_organization_list[organization_idx].step()

                poison_positions = np.where(np.isin(np.array(batch_idxs), selected_poison_indices.cpu().numpy()))[0]
                batch_num_poisoned = len(poison_positions)
                asr_batch = (predictions[poison_positions] == target_class).sum().item() / batch_num_poisoned if batch_num_poisoned > 0 else 0    
                
                total_asr += asr_batch
                # total_asr += 0  # placeholder if needed
            scheduler.step()
            for sched in scheduler_organization_list:
                sched.step()

            asr_epoch = total_asr/len(batch_idxs_list)

            train_auc_array.append(np.mean(train_auc_array_temp))
            train_loss = train_loss.detach().cpu().numpy()
            train_loss_array.append(train_loss.item())
            print("ASR for {0}-th epoch is {1}".format(i, asr_epoch))
            
            print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, train_loss, np.mean(train_auc_array_temp)))
            
            top_model.eval()
            for org_model in organization_models.values():
                org_model.eval()

            with torch.no_grad():
                # Ensure validation inputs are on GPU
                for organization_idx in range(organization_num):
                    X_val_vertical_FL[organization_idx] = X_val_vertical_FL[organization_idx].to(device)
                # y_val is already on device

                batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
                
                for batch_idxs in batch_idxs_list:
                    val_auc_array_temp = []
                    val_loss_array_temp = []
                    organization_outputs_for_val = {}
                    
                    # Compute outputs for each organization
                    for organization_idx in range(organization_num):
                        organization_outputs_for_val[organization_idx] = organization_models[organization_idx](X_val_vertical_FL[organization_idx][batch_idxs])
                    
                    # Concatenate organization outputs along the feature dimension
                    organization_outputs_for_val_cat = torch.cat([organization_outputs_for_val[j] for j in range(organization_num)], dim=1).float()
                    
                    # Forward pass through top model
                    log_probs = top_model(organization_outputs_for_val_cat)
                    val_loss = criterion(log_probs, y_val[batch_idxs].long())  # y_val already on GPU
                    predictions = torch.argmax(log_probs, dim=1)
                    
                    # Ensure both tensors for comparison are on the same device
                    correct = (predictions == y_val[batch_idxs].to(device)).sum().item()
                    total = y_val[batch_idxs].size(0)
                    val_auc = correct / total
                    val_auc_array_temp.append(val_auc)
                    val_loss_array_temp.append(val_loss.detach().cpu().numpy())
                
                val_auc_array.append(np.mean(val_auc_array_temp))
                val_loss_array.append(np.mean(val_loss_array_temp))
                print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
                print("learning_rates: ", learning_rates)
        
                # print(time.time()-or)
        top_model.eval()
        for org in organization_models.values():
            org.eval()

        # Clone the second-half training data for trigger insertion (to avoid modifying the originals)
        triggered_train_second = X_train_vertical_FL[1].clone()

        # Identify training indices that originally belong to the source class
        source_indices = (y_train == source_class).nonzero(as_tuple=True)[0]

        # Insert the trigger into these training images.
        # Here, we use a fixed insertion location for simplicity.
        fixed_r, fixed_c = 5, 5
        for idx in source_indices:
            triggered_train_second[idx] = insert_trigger_patch(
                triggered_train_second[idx].clone(), fixed_r, fixed_c, window_size, trigger_value
            )

        # Perform a forward pass on the training data with the trigger
        X_train_first = X_train_vertical_FL[0].to(device)
        triggered_train_second = triggered_train_second.to(device)
        with torch.no_grad():
            first_features_train = organization_models[0](X_train_first)
            second_features_train = organization_models[1](triggered_train_second)
            combined_features_train = torch.cat((first_features_train, second_features_train), dim=1)
            outputs_train = top_model(combined_features_train)
            predictions_train = torch.argmax(outputs_train, dim=1)

        # Compute Train ASR only on the triggered samples (i.e., those originally from source class)
        num_triggered_train = len(source_indices)
        print("num_triggered_train: ", num_triggered_train)
        num_successful_train = (predictions_train[source_indices] == target_class).sum().item()
        print("num_successful_train: ", num_successful_train)
        # num_successful_train = train_auc_array[-1]
        train_asr = num_successful_train / num_triggered_train
        print("Train ASR:", train_asr)


        triggered_val_second = X_val_vertical_FL[1].clone()
        source_val_indices = (y_val == source_class).nonzero(as_tuple=True)[0]
        fixed_r, fixed_c = 5, 5

        for idx in source_val_indices:
            r, c = fixed_r, fixed_c
            triggered_val_second[idx] = insert_trigger_patch(triggered_val_second[idx].clone(), r, c, window_size, trigger_val)

        X_val_first = X_val_vertical_FL[0].to(device)
        triggered_val_second = triggered_val_second.to(device)
        with torch.no_grad():
            first_features = organization_models[0](X_val_first)
            second_features = organization_models[1](triggered_val_second)
            combined_features = torch.cat((first_features, second_features), dim=1)
            outputs = top_model(combined_features)
            predictions = torch.argmax(outputs, dim=1)

        num_triggered = len(source_val_indices)
        num_successful = (predictions[source_val_indices] == target_class).sum().item()
        asr = num_successful / num_triggered

        print("test asr: ", asr)
        
        print("learning_rates: ", learning_rates)
        
        return train_loss_array, val_loss_array, train_auc_array, val_auc_array, 0  # asr_epoch placeholder

