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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader  # type: ignore
from utils import load_dat, batch_split
# Import CIFAR-100 models (assumed available)
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10
import random
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR100
import yaml
import copy
from p2vfl_utils import (split_tabular_data, concatenate_outputs, compute_contribution,
                         update_epsilons, compute_rewards, load_or_save_data,
                         load_or_save_vertical_split_data, batch_train_cifar10)
from set_environment_variables import set_environment

one_hot = False
fs  = False
contribution_measurement = True
dp_during_train = True
dp_during_val = False
dp_during_test = True
utility_threshold = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

with open('/configs/p2vfl.yaml', 'r') as file:
    config = yaml.safe_load(file)
 
# VFL parameters
data_type = 'original'
model_type = 'vertical'

delta = config['incentive']['delta']
sensitivity = config['incentive']['sensitivity']
alpha = config['incentive']['alpha']    # contribution term
beta = config['incentive']['beta']        # privacy term
client_costs_ratio = config['incentive']['client_costs_ratio']
total_tokens = config['incentive']['total_tokens']
client_actual_resources = config['incentive']['client_actual_resources']

default_organization_num = 3
one_hot = False

class Vertical_FL_Train:
    '''Vertical Federated Learning Training Class for CIFAR-100'''
    def __init__(self, active_clients):
        if active_clients is None:
            self.active_clients = [True] * (default_organization_num - 1)
        else:
            self.active_clients = active_clients

    # p2vfl gradient reward computation
    def grad_reward_epsilon(self, grad_lossdiff_emb, grad_noise, loss_diff, train_loss, epsilons, sensitivity, client_costs, alpha, beta):
        train_loss = torch.tensor(train_loss)
        client_costs_tensor = torch.tensor(client_costs, device=device)
        
        grad_lossdiff_emb = grad_lossdiff_emb.to(device)
        loss_diff = loss_diff.to(device)
        train_loss = train_loss.to(device)
        
        first_term = alpha * ((torch.sum(grad_lossdiff_emb * grad_noise) * 100 * (1+loss_diff) / train_loss) * client_costs_tensor**(1/5))
        second_term = beta * (-sensitivity / epsilons**2)
        return first_term + second_term
        
    def run(self, args, learning_rates, batch_size):
        '''Main function for CIFAR-100 training'''
        data_type = args.data_type                  
        epochs = args.epochs 
        organization_num = args.organization_num
        step_size_for_epsilon_feedback = args.step_size_for_epsilon_feedback
        active_clients = [True] * (organization_num + 1)
          
        total_parties = organization_num + 1
        # Set environment for CIFAR-100
        total_tokens, client_costs_ratio, client_actual_resources, num_warmup_epochs = set_environment(total_parties, 'CIFAR10')
        print("client_costs_ratio:", client_costs_ratio)
   
        attribute_split_array = np.zeros(total_parties).astype(int)  
        
        # p2vfl: initialize epsilons (privacy budgets)
        epsilons = []#*total_parties
        print("organization_num: ", total_parties)
        for _ in range(total_parties):
            epsilons.append(random.uniform(0.5, 1))
        print("epsilons initialization: ", epsilons)
        
        # Dataset preprocessing for CIFAR-100
        if data_type == 'original':
            if args.dname == 'CIFAR100':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor()
                    # Optionally add normalization if desired
                ])
                transform_val = transforms.Compose([
                    transforms.ToTensor()
                    # Optionally add normalization here as well
                ])
                train_set = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
                # For checking the data, get the first sample
                img, label = train_set[0]
                print(f"Min pixel value: {img.min()}, Max pixel value: {img.max()}")
                test_set = CIFAR100(root='./data', train=False, download=True, transform=transform_val)
                
                # Combine train and test datasets
                train_loader = DataLoader(train_set, batch_size=len(train_set))
                test_loader = DataLoader(test_set, batch_size=len(test_set))
                train_images, train_labels = next(iter(train_loader))
                test_images, test_labels = next(iter(test_loader))
                X = torch.cat((train_images, test_images), dim=0)
                y = torch.cat((train_labels, test_labels), dim=0)
        else:
            file_path = "./dataset/{0}.dat".format(args.dname)
            X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
        
        # Initialize arrays for tracking performance
        train_loss_array = []
        val_loss_array = []
        val_auc_array = []
        train_auc_array = []
        
        # Process the images: X shape is (N, 3, 32, 32)
        images_np = X.cpu().numpy()
        N = images_np.shape[0]
        
        base_width = 32 // total_parties
        remainder = 32 % total_parties
        widths = [base_width + (1 if i < remainder else 0) for i in range(total_parties)]
        
        image_parts_np = [
            np.zeros((N, 3, 32, widths[i]), dtype=np.float32)
            for i in range(total_parties)
        ]
        for n in range(N):
            current_col = 0
            for i in range(total_parties):
                end_col = current_col + widths[i]
                image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
                current_col = end_col
        encoded_vertical_splitted_data = image_parts_np

        print('X shape:', X.shape)
        random_seed = 1001
        
        # Split the vertically partitioned data into training and validation sets
        X_train_vertical_FL = {}
        X_val_vertical_FL = {}
    
        for organization_idx in range(total_parties):
            if organization_idx == 0:
                # The active party holds the labels.
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                y_train_val = y
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
                    train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=random_seed)
            else:
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                dummy_labels = np.zeros(len(X_train_val))
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
                    train_test_split(X_train_val, dummy_labels, test_size=0.20, random_state=random_seed)
        
        train_loader_list, val_loader_list = [], []
        for organization_idx in range(total_parties):
            X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
            X_val_vertical_FL[organization_idx] = torch.from_numpy(X_val_vertical_FL[organization_idx]).float()
            print("X_train_vertical_FL[{}]: ".format(organization_idx), X_train_vertical_FL[organization_idx].shape)
            print("X_val_vertical_FL[{}]: ".format(organization_idx), X_val_vertical_FL[organization_idx].shape)
            train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size, shuffle=True))
            val_loader_list.append(DataLoader(X_val_vertical_FL[organization_idx], batch_size=batch_size, shuffle=False))
        
        y_train = torch.from_numpy(y_train.numpy()).long()
        y_val = torch.from_numpy(y_val.numpy()).long()
        train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
        val_loader_list.append(DataLoader(y_val, batch_size=batch_size))
        
        # NN architecture setup: adjust top_output_dim for CIFAR-100 (100 classes)
        num_organization_hidden_units = 128
        organization_hidden_units_array = [np.array([num_organization_hidden_units]) for _ in range(total_parties)]
        organization_output_dim = np.array([64 for _ in range(total_parties)])
        top_hidden_units = np.array([512, 256, 64])
        top_output_dim = 100  # CIFAR-100 has 100 classes
        
        organization_models = {}
        for organization_idx in range(total_parties):
            organization_models[organization_idx] = torch_organization_model_cifar10(out_dim=organization_output_dim[organization_idx]).to(device)
        print("organization_output_dim: ", organization_output_dim)
        
        top_model = torch_top_model_cifar10(sum(organization_output_dim), top_hidden_units, top_output_dim)
        top_model = top_model.to(device).float()

        print('\nStart vertical FL on CIFAR-100......\n')
        criterion = nn.CrossEntropyLoss()
        top_model.train()

        top_models = {i: copy.deepcopy(top_model) for i in range(total_parties)}
        weights_and_biases = {i: top_models[i].state_dict() for i in range(total_parties)}

        ######################################################################################################
        # DEFINE OPTIMIZERS AND SCHEDULERS
        ######################################################################################################
        # Bottom (organization) models
        optimizer_bottom_models = {
            organization_idx: torch.optim.SGD(
                organization_models[organization_idx].parameters(),
                lr=learning_rates[1],  # e.g., set to 0.1
                momentum=0.9,
                weight_decay=1e-3
            )
            for organization_idx in range(total_parties)
        }
        scheduler_organization_list = [
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer_bottom_models[organization_idx],
                milestones=[15, 50, 75],
                gamma=args.g_bm
            )
            for organization_idx in range(total_parties)
        ]
        # Top model optimizer and scheduler
        top_model_optimizer = torch.optim.SGD(
            top_model.parameters(),
            lr=learning_rates[0],
            momentum=0.9,
            weight_decay=5e-4
        )
        top_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            top_model_optimizer,
            milestones=[15, 50, 75],
            gamma=args.g_tm
        )
        # Individual top models for contribution evaluation
        top_optimizer_individual_model = {
            organization_idx: torch.optim.SGD(
                top_models[organization_idx].parameters(),
                lr=learning_rates[0],
                momentum=0.9,
                weight_decay=5e-4
            )
            for organization_idx in range(total_parties)
        }
        top_scheduler_individual_model = [
            torch.optim.lr_scheduler.MultiStepLR(
                top_optimizer_individual_model[organization_idx],
                milestones=[15, 85],
                gamma=args.g_tm
            )
            for organization_idx in range(total_parties)
        ]
        ######################################################################################################
        # END OPTIMIZERS AND SCHEDULERS
        ######################################################################################################

        contributions_array = []
        rewards_array = []
        epsilons_array = []
        utility_array = []
        contribution_term_array = []
        privacy_term_array = []
        reward_distribution_array = []
                
        for i in range(epochs):
            contribution_per_organization = {}
            st = time.time()
            print('Epoch: ', i)
            batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
            grad_reward_eps = [0 for _ in range(total_parties)]  # p2vfl
            
            for bidx in range(len(batch_idxs_list)):
                grad_lossdiff_emb = {} 
                batch_idxs = batch_idxs_list[bidx]
                train_auc_array_temp = []
                train_loss_array_temp = []

                top_model.zero_grad()
                for organization_idx in range(total_parties):
                    optimizer_bottom_models[organization_idx].zero_grad()  # Zero gradients for bottom model
                    
                organization_outputs = {}
                organization_outputs_cat, organization_outputs, train_grad_noise = concatenate_outputs(
                    total_parties, organization_models, batch_idxs, X_train_vertical_FL, active_clients, dp_during_train, epsilons, delta, sensitivity, bidx
                )
                organization_outputs_cat = organization_outputs_cat.float()
                
                outputs = top_model(organization_outputs_cat)
                y_train = y_train.to(device)
                log_probs = outputs
                train_loss = criterion(log_probs, y_train[batch_idxs])
                predictions = torch.argmax(log_probs, dim=1)
                correct = (predictions == y_train[batch_idxs]).sum().item()
                total = y_train[batch_idxs].size(0)
                    
                train_loss.backward()  # Backpropagation
                train_auc_array_temp.append(correct / total)
                train_loss_array_temp.append(train_loss.detach().cpu().item())

                torch.nn.utils.clip_grad_norm_(top_model.parameters(), max_norm=1.0)
                
                inputs = {} 
                for key in range(total_parties):
                    temp = []
                    for k in organization_outputs:
                        if k == key:
                            temp.append(torch.zeros_like(organization_outputs[k]))
                        else:
                            temp.append(organization_outputs[k])
                    temp_tensor = torch.cat(temp, dim=1)
                    inputs[key] = temp_tensor

                contribution_per_organization = compute_contribution(
                    total_parties, inputs, top_model, criterion, y_train, batch_idxs, 
                    top_model_optimizer, weights_and_biases, train_loss, contribution_per_organization
                )
                
                top_model_optimizer.step()
                for organization_idx in range(total_parties):
                    optimizer_bottom_models[organization_idx].step()  # Bottom model update
                                    
            top_model_scheduler.step()
            for sched in scheduler_organization_list:
                sched.step()  # Bottom model schedulers
            for sched in top_scheduler_individual_model:
                sched.step()

            train_loss_detached = train_loss.detach().cpu().numpy()
            
            # Compute gradient reward epsilon
            for organization_idx in range(total_parties):
                if active_clients[organization_idx]:
                    grad_lossdiff_emb[organization_idx] = organization_outputs[organization_idx].grad
                else:
                    grad_lossdiff_emb[organization_idx] = 0
                
            for organization_idx in range(total_parties):  
                grad_reward_eps_batch = self.grad_reward_epsilon(
                    -1 * grad_lossdiff_emb[organization_idx],
                    train_grad_noise,
                    contribution_per_organization[organization_idx]['average'],
                    train_loss_detached,
                    epsilons[organization_idx], 
                    sensitivity, 
                    client_costs_ratio[organization_idx],
                    alpha,
                    beta
                )
                grad_reward_eps[organization_idx] += grad_reward_eps_batch
                            
            num_batches = len(batch_idxs_list)
            for organization_idx in range(total_parties):
                grad_reward_eps[organization_idx] /= num_batches

            print("epsilons: ", epsilons)
            epsilons = update_epsilons(total_parties, active_clients, step_size_for_epsilon_feedback, grad_reward_eps, epsilons)
            c_temp = [contribution_per_organization[key]['average'] for key in range(total_parties)]
            cons = np.array([round(float(x), 3) for x in c_temp])
            contributions_array.append(c_temp)
            contribution_term, privacy_term, rewards, reward_distribution, client_utility, reward_distribution_array = compute_rewards(
                cons, client_costs_ratio, sensitivity, epsilons, alpha, beta, total_tokens, total_parties, reward_distribution_array, client_actual_resources
            )
            
            contribution_term_array.append(contribution_term)
            privacy_term_array.append(privacy_term)
            rewards_array.append(rewards)
            utility_array.append(client_utility)
            epsilons_array.append(epsilons.copy())
            print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, np.mean(train_loss_array_temp), np.mean(train_auc_array_temp)))
            train_auc_array.append(np.mean(train_auc_array_temp))
            train_loss = train_loss.detach().cpu().numpy()
            train_loss_array.append(train_loss_array_temp[0])
                
            if (i+1) % 1 == 0:
                top_model.eval()
                for org_model in organization_models.values():
                    org_model.eval()
                with torch.no_grad():
                    for organization_idx in range(total_parties):
                        X_val_vertical_FL[organization_idx] = X_val_vertical_FL[organization_idx].to(device)
                    y_val = y_val.to(device)
                    batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
                
                    for batch_idxs in batch_idxs_list:
                        val_auc_array_temp = []
                        val_loss_array_temp = []
                        organization_outputs_for_val = {}
                        
                        feature_mask_tensor_list = []
                        for organization_idx in range(total_parties):
                            organization_outputs_for_val[organization_idx] = organization_models[organization_idx](X_val_vertical_FL[organization_idx][batch_idxs])
                            feature_mask_tensor_list.append(torch.full(organization_outputs_for_val[organization_idx].shape, organization_idx))
                        organization_outputs_for_val_cat = organization_outputs_for_val[0].float()
                    
                        if len(organization_outputs_for_val) >= 2:
                            for organization_idx in range(1, total_parties):
                                organization_outputs_for_val_cat = torch.cat((organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1).float()

                        organization_outputs_for_val_cat = organization_outputs_for_val_cat.float()
                        log_probs = top_model(organization_outputs_for_val_cat)
                        val_loss = criterion(log_probs, y_val[batch_idxs].to(device).long())
                        predictions = torch.argmax(log_probs, dim=1)
                        correct = (predictions == y_val[batch_idxs]).sum().item()
                        total = y_val[batch_idxs].size(0)
                        val_auc_array_temp.append(correct / total)
                        val_loss_array_temp.append(val_loss.detach().cpu().item())
                    
                    print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
                    val_auc_array.append(np.mean(val_auc_array_temp))
                    val_loss_array.append(np.mean(val_loss_array_temp))
                    time_taken = time.time() - st
                    print("time_taken for one epoch= ", time_taken)
        print("train_auc_array =", train_auc_array)
        print("train_loss_array=", train_loss_array)
        print('val_auc_array=', val_auc_array)
        print("val_loss_array=", val_loss_array)
        print("learning_rates: ", learning_rates)

        print("contribution_term_array", [arr.tolist() for arr in contribution_term_array])
        print("rewards_array", [arr.tolist() for arr in rewards_array])
        print("utility_array", [arr.tolist() for arr in utility_array])
        print("epsilons_array", epsilons_array)
        print("learning_rates: ", learning_rates)

        return train_loss_array, val_loss_array, train_auc_array, val_auc_array, time_taken
