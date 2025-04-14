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
from torch.utils.data import DataLoader # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10
import numpy as np
import random
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from feature_inference_attack import Generator

# from pytorchtools import EarlyStopping
# early_stopping = EarlyStopping(patience=5, verbose=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
default_organization_num = 3


one_hot = False
class Vertical_FL_Train:
    ''' Vertical Federated Learning Training Class'''
    def __init__(self, active_clients):
        if active_clients is None:
            self.active_clients = [True] * (default_organization_num - 1)
        else:
            self.active_clients = active_clients
        
    def run(self, args, learning_rates, batch_size):

        ''' Main function for the program'''
        data_type = args.data_type                  
        epochs = args.epochs 
        

        print("num_default_epochs: ", epochs)
        
        organization_num = args.organization_num    
        attribute_split_array = \
            np.zeros(organization_num).astype(int)  
        
        # dataset preprocessing
        if data_type == 'original':
            if args.dname == 'CIFAR10':
                
                transform_train = transforms.Compose([
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomRotation(15),
                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor()
                        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                    ])
                transfor_val = transforms.Compose([
                        transforms.ToTensor()
                        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                    ])
                train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
                img, label = train_set[0]  # img is a torch tensor

                # Check min/max values
                # print(f"Min pixel value: {img.min()}, Max pixel value: {img.max()}")
                
                test_set = CIFAR10(root='./data', train=False, download=True, transform=transfor_val)
                
                # Combine train and test datasets
                train_loader = DataLoader(train_set, len(train_set))
                test_loader = DataLoader(test_set, len(test_set))
                
                train_images, train_labels = next(iter(train_loader))
                test_images, test_labels = next(iter(test_loader))
                
                # Concatenate train and test
                X = torch.cat((train_images, test_images), dim=0)
                y = torch.cat((train_labels, test_labels), dim=0)

                print("y[:10]", y[:10])

                samples_per_class = 6000  # change this number as needed

                # Convert labels to numpy for easier processing
                y_np = y.numpy()
                balanced_indices = []

                num_classes = 10  # CIFAR-10 has 10 classes
                for cls in range(num_classes):
                    # Find all indices for the current class
                    cls_indices = np.where(y_np == cls)[0]
                    # Shuffle indices to randomize
                    np.random.shuffle(cls_indices)
                    # Select the desired number of samples per class
                    selected = cls_indices[:samples_per_class]
                    balanced_indices.extend(selected.tolist())

                balanced_indices = np.array(balanced_indices)
                print("Total balanced subset size:", len(balanced_indices))

                # ---------------------------
                # Option 1: Create Tensors Directly
                # ---------------------------
                X_balanced = X[balanced_indices]
                y_balanced = y[balanced_indices]

                print("Balanced X shape:", X_balanced.shape)
                print("Balanced y shape:", y_balanced.shape)
    
        else:
            file_path = "./dataset/{0}.dat".format(args.dname)
            X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
        
    
       
        # initialize the arrays to be return to the main function
        train_loss_array = []
        val_loss_array = []
        val_auc_array = []
        train_auc_array = []

        X = X_balanced
        y = y_balanced

        
        images_np = X.cpu().numpy()  # shape: (N, 3, 32, 32)
        N = images_np.shape[0]
        
        base_width = 32 // organization_num
        remainder = 32 % organization_num
        widths = [base_width + (1 if i < remainder else 0) for i in range(organization_num)]
        
        image_parts_np = [
            np.zeros((N, 3, 32, widths[i]), dtype=np.float32)
            for i in range(organization_num)
        ]
        
        for n in range(N):
            current_col = 0
            for i in range(organization_num):
                end_col = current_col + widths[i]
                image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
                current_col = end_col
        
        encoded_vertical_splitted_data = image_parts_np

        x_adv_new_np = encoded_vertical_splitted_data[0]  # This is a numpy array with shape: (N, 3, 32, widths[0])
        x_adv_new = torch.from_numpy(x_adv_new_np).float()  # Now shape: (N, 3, 32, widths[0])
        torch.save(x_adv_new, 'x_adv_new.pt')


        print('X shape:',X.shape)
        # set up the random seed for dataset split
        random_seed = 1001
        
        # split the encoded data samples into training and test datasets
        X_train_vertical_FL = {}
        X_val_vertical_FL = {}
        # X_test_vertical_FL = {}
        # selected_features = None

        for organization_idx in range(organization_num):
            if organization_idx == 0:
                # For the active party, use the labels.
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                y_train_val = y
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
                    train_test_split(X_train_val, y_train_val, test_size=10000/60000,random_state=random_seed)
                #, random_state=random_seed
            else:
                # For other parties, we don't have labels so use dummy labels.
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                dummy_labels = np.zeros(len(X_train_val))
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
                    train_test_split(X_train_val, dummy_labels, test_size=10000/60000,random_state=random_seed)
        
        
        train_loader_list, test_loader_list, val_loader_list = [], [], []
        for organization_idx in range(organization_num):
        
            X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
            X_val_vertical_FL[organization_idx] = torch.from_numpy(X_val_vertical_FL[organization_idx]).float()
            # X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()

            print("X_train_vertical_FL[organization_idx]: ", X_train_vertical_FL[organization_idx].shape)
            print("X_val_vertical_FL[organization_idx]: ", X_val_vertical_FL[organization_idx].shape)
            # print("X_test_vertical_FL[organization_idx]: ", X_test_vertical_FL[organization_idx].shape)
            

            train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size, shuffle=True))
            val_loader_list.append(DataLoader(X_val_vertical_FL[organization_idx], batch_size=batch_size, shuffle=False))
            # test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
            
        y_train = torch.from_numpy(y_train.numpy()).long()
        y_val = torch.from_numpy(y_val.numpy()).long()
        # y_test = torch.from_numpy(y_test.numpy()).long()
        
        train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
        val_loader_list.append(DataLoader(y_val, batch_size=batch_size))
        # test_loader_list.append(DataLoader(y_test, batch_size=len(X_test_vertical_FL[organization_idx])))
        
        # NN architecture
        #num_organization_hidden_layers = params['num_organization_hidden_layers']
        num_organization_hidden_units = 128   #params['num_organization_hidden_units']
        organization_hidden_units_array = [np.array([num_organization_hidden_units])]*organization_num   #* num_organization_hidden_layers
        organization_output_dim = np.array([64 for i in range(organization_num)])
        num_top_hidden_units = 64  #params['num_top_hidden_units']
        # top_hidden_units = np.array([64,64])
        top_hidden_units = np.array([512,256,64])
        top_output_dim = 10
        
        organization_models = {}
        # build the client models
        # for organization_idx in range(organization_num):
        # 			organization_models[organization_idx] = \
        # 				torch_organization_model_cifar10(organization_output_dim[organization_idx]).to(device)
                    
        for organization_idx in range(organization_num):
                    organization_models[organization_idx] = \
                        torch_organization_model_cifar10(out_dim=organization_output_dim[organization_idx]).to(device)
        
        print("organization_output_dim: ", organization_output_dim)
        # build the top model over the client models
        top_model = torch_top_model_cifar10(64*(organization_num), top_hidden_units, top_output_dim)
        # Suppose self.active_clients is defined so that its length equals (organization_num - 1)
        
        top_model = top_model.to(device).float()

         
        optimizer = torch.optim.SGD(
        top_model.parameters(),
        lr=learning_rates[0],       # e.g., set to 0.1 for CIFAR-10
        momentum=0.9,
        weight_decay=5e-4          # typical weight decay for CIFAR-10
        )

        # Update bottom (organization) model optimizers
        optimizer_organization_list = []
        for organization_idx in range(organization_num):
            if organization_idx == 1:
                lr_for_client = learning_rates[1]   # e.g., reduce by 50%
            else:
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
            gamma=args.g_tm #0.0001
        )

        scheduler_organization_list = [
            torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 30, 40, 60, 80], gamma=args.g_bm)
            for opt in optimizer_organization_list
        ]  
        
        print('\nStart vertical FL......\n')   
        criterion = nn.CrossEntropyLoss()
        top_model.train()
        
        for i in range(epochs):
            print('Epoch: ', i)
            print("Current top model LR:", optimizer.param_groups[0]['lr'])
            batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
            total_asr = 0
            total_misclassified = 0
            for bidx in range(len(batch_idxs_list)):
                batch_idxs = batch_idxs_list[bidx]
                train_auc_array_temp=[]
                optimizer.zero_grad()
                for organization_idx in range(organization_num):
                    optimizer_organization_list[organization_idx].zero_grad()

                
                organization_outputs = {}
                                    
                for organization_idx in range(organization_num):
                        data_batch = X_train_vertical_FL[organization_idx][batch_idxs].to(device)
                            # Normal forward for other clients
                        organization_outputs[organization_idx] = organization_models[organization_idx](data_batch)
                    
                organization_outputs_cat = organization_outputs[0].float()
                for organization_idx in range(1, organization_num):
                    if self.active_clients[organization_idx - 1]:
                        organization_outputs_cat = torch.cat((organization_outputs_cat, organization_outputs[organization_idx]), 1).float()

                    else:
                        if bidx == 0:
                            print("client ", organization_idx-1, " input zeroed out")
                        zeroed_inputs = torch.zeros_like(organization_outputs[organization_idx])
                        organization_outputs_cat = torch.cat((organization_outputs_cat, zeroed_inputs), 1).float()
                                        
                
                organization_outputs_cat = organization_outputs_cat.float()

                
                outputs = top_model(organization_outputs_cat)
                y_train = y_train.to(device)
                log_probs = outputs
                train_loss = criterion(log_probs, y_train[batch_idxs])
                train_loss.backward()
                predictions = torch.argmax(log_probs, dim=1)
                correct = (predictions == y_train[batch_idxs]).sum().item()
                batch_misclassified = (predictions == target_class)
                total = y_train[batch_idxs].size(0)
                train_auc = correct / total
                train_auc_array_temp.append(train_auc)
                batch_misclassified = (predictions == target_class).sum().item()
                total_misclassified += batch_misclassified
                
                torch.nn.utils.clip_grad_norm_(top_model.parameters(), max_norm=1.0)

            
                optimizer.step() # adjust parameters based on the calculated gradients 
                
                for organization_idx in range(organization_num):
                    # if self.active_clients[organization_idx - 1]:
                        optimizer_organization_list[organization_idx].step()

                
            scheduler.step()
            for sched in scheduler_organization_list:
                sched.step()

            asr_epoch = total_asr/len(batch_idxs_list)
            
            print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, train_loss.detach().cpu().numpy(), np.mean(train_auc_array_temp)))
            train_auc_array.append(np.mean(train_auc_array_temp))
            train_loss=train_loss.detach().cpu().numpy()
            train_loss_array.append(train_loss.item())
                
            if (i+1)%1 == 0:
        
                top_model.eval()
                for org_model in organization_models.values():
                    org_model.eval()

                with torch.no_grad():

                    for organization_idx in range(organization_num):
                            X_val_vertical_FL[organization_idx] = X_val_vertical_FL[organization_idx].to(device)
                            # X_test_vertical_FL[organization_idx] = X_test_vertical_FL[organization_idx].to(device)
                    y_val = y_val.to(device)
                    # y_test = y_test.to(device)

                    batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
                
                    for batch_idxs in batch_idxs_list:
                        val_auc_array_temp = []
                        val_loss_array_temp = []
                        organization_outputs_for_val = {}
                        
                        feature_mask_tensor_list = []
                        for organization_idx in range(organization_num):
                            organization_outputs_for_val[organization_idx] = organization_models[organization_idx](X_val_vertical_FL[organization_idx][batch_idxs])
                            feature_mask_tensor_list.append(torch.full(organization_outputs_for_val[organization_idx].shape, organization_idx))
                        organization_outputs_for_val_cat = organization_outputs_for_val[0].float()
                    
                        #DP
                        if len(organization_outputs_for_val) >= 2:
                            for organization_idx in range(1, organization_num):
                                organization_outputs_for_val_cat = torch.cat((organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1).float()
                        
                        organization_outputs_for_val_cat = organization_outputs_for_val_cat.float()

                        log_probs = top_model(organization_outputs_for_val_cat)
                        val_loss = criterion(log_probs, y_val[batch_idxs].to(device).long())
                        predictions = torch.argmax(log_probs, dim=1)
                        correct = (predictions == y_val[batch_idxs]).sum().item()
                        total = y_val[batch_idxs].size(0)
                        val_auc = correct / total
                        val_auc_array_temp.append(val_auc)
                        val_loss_array_temp.append(val_loss.detach().cpu().numpy())
            val_auc_array.append(np.mean(val_auc_array_temp))
            val_loss_array.append(np.mean(val_loss_array_temp))
            print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))

        print("learning_rates: ", learning_rates)

        # === GRN Step 1: Collect (x_adv, v) ===

        torch.save(top_model.state_dict(), "top_model.pt")
        print("Saved top_model.pt")

        # Save each organization model's state dictionary
        for org_idx, model in organization_models.items():
            torch.save(model.state_dict(), f"organization_model_{org_idx}.pt")
            print(f"Saved organization_model_{org_idx}.pt")
        
        print("\n[GRN] Collecting (x_adv, v) for adversary (client 0)...")

        x_adv_list = []
        v_list = []

        top_model.eval()
        for org_model in organization_models.values():
            org_model.eval()

        batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)

        for batch_idxs in batch_idxs_list:
            with torch.no_grad():
                # Get attacker's known input (client 0)
                x_adv_batch = X_val_vertical_FL[0][batch_idxs].to(device)
                adv_output = organization_models[0](x_adv_batch)

                # Get the rest of the clients' outputs
                org_out_cat = adv_output
                for organization_idx in range(1, organization_num):
                    client_output = organization_models[organization_idx](
                        X_val_vertical_FL[organization_idx][batch_idxs].to(device)
                    )
                    org_out_cat = torch.cat((org_out_cat, client_output), dim=1)

                # Get predictions from top model
                v_batch = top_model(org_out_cat)

                # Save attacker's input and corresponding predictions
                x_adv_list.append(x_adv_batch.detach().cpu())
                v_list.append(v_batch.detach().cpu())

        # Combine all batches into a single tensor
        x_adv_all = torch.cat(x_adv_list, dim=0)
        v_all = torch.cat(v_list, dim=0)

        torch.save(x_adv_all, 'x_adv_all.pt')
        torch.save(v_all,'v_all.pt')

        print("[GRN] Collected x_adv_all shape:", x_adv_all.shape)
        print("[GRN] Collected v_all shape:", v_all.shape)

        return train_loss_array, val_loss_array, train_auc_array, val_auc_array