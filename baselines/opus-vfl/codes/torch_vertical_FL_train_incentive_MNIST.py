import time
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../torch_utils/')
from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader  # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_organization_model, torch_top_model
import torch.nn.functional as F
from torch_generate_noise import add_Gaussian_noise
from set_environment_variables import set_environment
import random
import argparse
import torch.nn.functional as F
import copy
import yaml
import argparse

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

one_hot = False

def load_config(config_path="../../configs/p2vfl.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()

data_type = 'original'
model_type = 'vertical'
dname = config['experiment']['dataset']
num_iterations = config['experiment']['iterations']
# incentive parameters
data_type = 'original'
model_type = "vertical"
delta = config['incentive']['delta']
sensitivity = config['incentive']['sensitivity']

alpha = 1   
beta = 1
dname = 'MNIST'
batch_size = config['experiment']['batch_size']
step_size_for_epsilon_feedback = 0.1
dp_during_train  = True

class Vertical_FL_Train:
    ''' Vertical Federated Learning Training Class'''
    def __init__(self, active_clients):
        # If active_clients is not provided, assume a default length of 6 organizations (aggregator/active party + 5 passive parties)
        if active_clients is None:
            self.active_clients = [True] * 15
        else:
            self.active_clients = active_clients

    def grad_reward_epsilon(self, grad_lossdiff_emb, grad_noise, loss_diff, train_loss, epsilons, sensitivity, client_costs):
        first_term = (torch.sum(grad_lossdiff_emb * grad_noise) * 100 * (1+loss_diff) / train_loss) * client_costs**(1/2)
        second_term = -sensitivity / epsilons**2
        return first_term + second_term

    def distribute_rewards(self, rewards, total_tokens, num_clients):
        total_rewards = np.ceil(
            rewards / np.sum(rewards)
            * (num_clients * (num_clients + 1) / 2)
        )
        total_tokens -= total_rewards
        print("total_tokens: ", len(total_tokens))
        print("total_rewards: ", len(total_rewards))
        return total_tokens, total_rewards
    
    def compute_snr(self, signal: torch.Tensor, noisy_signal: torch.Tensor) -> float:
        # Compute the noise as the difference between noisy and original signals.
        noise = noisy_signal - signal

        # Compute L2 norms for signal and noise.
        signal_norm = torch.norm(signal, p=2)
        noise_norm = torch.norm(noise, p=2)

        # Avoid division by zero by returning infinity if noise_norm is zero.
        if noise_norm.item() == 0:
            return float('inf')

        # Compute SNR in dB.
        snr_value = 20.0 * torch.log10(signal_norm / noise_norm)
        return snr_value.item()

    def run(self, args, learning_rates, batch_size):
        ''' Main function for the program'''
        data_type = args.data_type
        model_type = args.model_type
        epochs = args.epochs
        
        # organization_num is the number of passive parties.
        organization_num = args.organization_num  
        # FIXED: use exactly organization_num as total orgs (no extra aggregator)
        default_organization_num = organization_num  # was (organization_num + 1)

        # Environment settings – aggregator’s tokens are not considered in client_total_resources_for_training and client_costs_ratio
        total_tokens, client_costs_ratio, client_total_resources_for_training, num_warmup_epochs = set_environment(organization_num, 'MNIST')
        total_tokens = [10, 40, 50, 30, 60]
        # Create attribute_split_array for total organizations.
        attribute_split_array = np.zeros(default_organization_num).astype(int)
        
        # Initialize epsilons for all organizations.
        # All are passive parties, each with random epsilon values between 0.5 and 1.
        epsilons = [10000]
        print("organization_num (passive parties): ", organization_num+1)
        for _ in range(int(organization_num)+1):
            random_number = random.uniform(0.5, 1)
            epsilons.append(random_number)
        
        print("Initial epsilons: ", epsilons)
        
        # Dataset preprocessing
        if data_type == 'original':
            if args.dname == 'MNIST' or args.dname == 'FMNIST':
                file_path = "../../datasets/MNIST/{0}.csv".format(args.dname)
                X = pd.read_csv(file_path)
                y = X['class']
                X = X.drop(['class'], axis=1)
                
                N, dim = X.shape
                columns = list(X.columns)
                
                # Split attributes equally across all organizations
                attribute_split_array = np.ones(len(attribute_split_array)).astype(int) * int(dim / default_organization_num)
                if np.sum(attribute_split_array) > dim:
                    print('unknown error in attribute splitting!')
                elif np.sum(attribute_split_array) < dim:
                    missing_attribute_num = dim - np.sum(attribute_split_array)
                    attribute_split_array[-1] += missing_attribute_num
                else:
                    print('Successful attribute split for multiple organizations')
        else:
            file_path = "./dataset/{0}.dat".format(args.dname)
            X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)
        
        train_loss_array = []
        val_loss_array = []
        val_auc_array = []
        train_auc_array = []
        
        if model_type == 'vertical':
            # Define attributes per organization
            attribute_groups = []
            attribute_start_idx = 0
            for organization_idx in range(default_organization_num):
                attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
                attribute_groups.append(columns[attribute_start_idx: attribute_end_idx])
                attribute_start_idx = attribute_end_idx
            
            for organization_idx in range(default_organization_num):
                print('The number of attributes held by Organization {0}: {1}'.format(
                    organization_idx, len(attribute_groups[organization_idx])))
            
            # Get vertically splitted data
            vertical_splitted_data = {}
            encoded_vertical_splitted_data = {}
            
            chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            for organization_idx in range(default_organization_num):
                vertical_splitted_data[organization_idx] = X[attribute_groups[organization_idx]].values
                encoded_vertical_splitted_data[organization_idx] = chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
            
            if one_hot:
                chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
                for organization_idx in range(default_organization_num):
                    vertical_splitted_data[organization_idx] = X[attribute_groups[organization_idx]].values
                    encoded_vertical_splitted_data[organization_idx] = chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
            else:
                for organization_idx in range(default_organization_num):
                    vertical_splitted_data[organization_idx] = X[attribute_groups[organization_idx]].values
                encoded_vertical_splitted_data = vertical_splitted_data
            
            print('X shape:', X.shape)
            random_seed = 1001
            
            # Split each organization's data into train/val/test
            X_train_vertical_FL = {}
            X_val_vertical_FL = {}
            X_test_vertical_FL = {}
            
            for organization_idx in range(default_organization_num):
                test_set_size = 5
                # We'll still store labels with org0 for convenience
                if organization_idx == 0:
                    X_test_vertical_FL[organization_idx] = encoded_vertical_splitted_data[organization_idx][-test_set_size:]
                    y_test = y[-test_set_size:]
                    
                    X_train_val = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
                    y_train_val = y[:-test_set_size]
                    
                    X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
                        train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random_seed)
                else:
                    X_test_vertical_FL[organization_idx] = encoded_vertical_splitted_data[organization_idx][-test_set_size:]
                    X_train_val = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
                    dummy_labels = np.zeros(len(X_train_val))
                    
                    X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
                        train_test_split(X_train_val, dummy_labels, test_size=0.25, random_state=random_seed)
            
            train_loader_list, test_loader_list, val_loader_list = [], [], []
            for organization_idx in range(default_organization_num):
                X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
                X_val_vertical_FL[organization_idx] = torch.from_numpy(X_val_vertical_FL[organization_idx]).float()
                X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()
                
                train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size))
                val_loader_list.append(DataLoader(X_val_vertical_FL[organization_idx], batch_size=len(X_val_vertical_FL[organization_idx]), shuffle=False))
                test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
            
            y_train = torch.from_numpy(y_train.to_numpy()).long()
            y_val = torch.from_numpy(y_val.to_numpy()).long()
            y_test = torch.from_numpy(y_test.to_numpy()).long()
            train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
            val_loader_list.append(DataLoader(y_val, batch_size=batch_size))
            test_loader_list.append(DataLoader(y_test, batch_size=batch_size))
            
            # NN architecture
            num_organization_hidden_units = 128
            organization_hidden_units_array = [np.array([num_organization_hidden_units])] * default_organization_num
            # Each org outputs 64 dims → total input to top model is 64 * default_organization_num
            organization_output_dim = np.array([64 for _ in range(default_organization_num)])
            num_top_hidden_units = 64
            top_hidden_units = np.array([num_top_hidden_units])
            top_output_dim = 10
            
            # Build client models
            organization_models = {}
            for organization_idx in range(default_organization_num):
                organization_models[organization_idx] = torch_organization_model(
                    X_train_vertical_FL[organization_idx].shape[-1],
                    organization_hidden_units_array[organization_idx],
                    organization_output_dim[organization_idx]
                )
            
            # Build the top model using the concatenation of all organization outputs
            top_model = torch_top_model(np.sum(organization_output_dim), top_hidden_units, top_output_dim)
            
            optimizer = torch.optim.Adam(top_model.parameters(), lr=learning_rates[0], weight_decay=1e-5)
            optimizer_organization_list = []
            for organization_idx in range(default_organization_num):
                optimizer_organization_list.append(
                    torch.optim.Adam(organization_models[organization_idx].parameters(), lr=learning_rates[1], weight_decay=1e-5)
                )
            
            print('\nStart vertical FL......\n')
            criterion = nn.CrossEntropyLoss()
            top_model.train()
            
            top_models = {i: copy.deepcopy(top_model) for i in range(default_organization_num)}
            weights_and_biases = {i: top_models[i].state_dict() for i in range(default_organization_num)}
            optimizer_individual_model = {
                i: torch.optim.Adam(top_models[i].parameters(), learning_rates[0]) for i in range(default_organization_num)
            }
            
            contributions_array = []
            rewards_array = []
            epsilons_array = []
            utility_array = []
            contribution_term_array = []
            privacy_term_array = []
            reward_distribution_array = []
            time_per_epoch_array = []
            
            top_model.train()
            print(client_costs_ratio)
            print(client_total_resources_for_training)

            # Overwrite epochs if you want
            num_warmup_epochs = 0
            epochs = 60
                
            for i in range(epochs):
                contribution_per_organization = {}
                print('Epoch: ', i)
                st = time.time()
                batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
                grad_reward_eps = [0 for _ in range(default_organization_num)]
                
                for bidx in range(len(batch_idxs_list)):
                    grad_lossdiff_emb = {}
                    batch_idxs = batch_idxs_list[bidx]
                    train_auc_array_temp = []
                    
                    optimizer.zero_grad()
                    for organization_idx in range(default_organization_num):
                        optimizer_organization_list[organization_idx].zero_grad()
                    
                    organization_outputs = {}
                    for organization_idx in range(default_organization_num):
                        organization_outputs[organization_idx] = organization_models[organization_idx](
                            X_train_vertical_FL[organization_idx][batch_idxs]
                        )
                        organization_outputs[organization_idx].retain_grad()
                    
                    organization_outputs_cat = None
                    train_grad_noise = None  # track the last noise gradient
                    for organization_idx in range(default_organization_num):
                        if dp_during_train:
                            train_noise, temp_grad_noise = add_Gaussian_noise(
                                organization_outputs[organization_idx],
                                epsilons[organization_idx],
                                delta,
                                sensitivity
                            )
                            if organization_outputs_cat is None:
                                organization_outputs_cat = train_noise.float()
                            else:
                                organization_outputs_cat = torch.cat(
                                    (organization_outputs_cat, train_noise), 1
                                ).float()
                            noise = train_noise - organization_outputs[organization_idx]
                            snr_batch = self.compute_snr(organization_outputs[organization_idx], noise)
                            train_grad_noise = temp_grad_noise
                        else:
                            if organization_outputs_cat is None:
                                organization_outputs_cat = organization_outputs[organization_idx].float()
                            else:
                                organization_outputs_cat = torch.cat(
                                    (organization_outputs_cat, organization_outputs[organization_idx]), 1
                                ).float()
                    
                    outputs = top_model(organization_outputs_cat)
                    log_probs = outputs
                    train_loss = criterion(log_probs, y_train[batch_idxs])
                    predictions = torch.argmax(log_probs, dim=1)
                    correct = (predictions == y_train[batch_idxs]).sum().item()
                    total = y_train[batch_idxs].size(0)
                    train_auc = correct / total
                    train_auc_array_temp.append(train_auc)
                    
                    train_loss.backward()
                    
                    # Evaluate contributions for all organizations
                    inputs = {}
                    for key in range(1, default_organization_num):
                        temp = []
                        for k in organization_outputs:
                            if k == key:
                                temp.append(torch.zeros_like(organization_outputs[k]))
                            else:
                                temp.append(organization_outputs[k])
                        temp_tensor = torch.cat(temp, dim=1)
                        inputs[key] = temp_tensor.detach()
                    
                    for key in range(default_organization_num):
                        outputs_2 = top_models[key](inputs[key]) if key in top_models else top_model(inputs[key])
                        train_loss_without = criterion(outputs_2, y_train[batch_idxs].type(torch.LongTensor))
                        
                        if key in optimizer_individual_model:
                            optimizer_individual_model[key].zero_grad()
                            train_loss_without.backward()
                            optimizer_individual_model[key].step()
                            weights_and_biases[key] = top_models[key].state_dict() if key in top_models else top_model.state_dict()
                        
                        if train_loss.item() != 0:
                            net_contribution = ((train_loss_without.item() - train_loss.item()) * 100) / train_loss
                        else:
                            net_contribution = train_loss_without.item() * 100
                        
                        if key in contribution_per_organization:
                            contribution_per_organization[key]['sum'] += net_contribution
                            contribution_per_organization[key]['count'] += 1
                            contribution_per_organization[key]['average'] = \
                                contribution_per_organization[key]['sum'] / contribution_per_organization[key]['count']
                        else:
                            contribution_per_organization[key] = {
                                'sum': net_contribution,
                                'count': 1,
                                'average': net_contribution
                            }
                    
                    for organization_idx in range(default_organization_num):
                        optimizer_organization_list[organization_idx].step()
                    
                    optimizer.step()
                    for organization_idx in range(default_organization_num):
                        optimizer_organization_list[organization_idx].step()
                
                # Gather gradients for all organizations
                for organization_idx in range(default_organization_num):
                    grad_lossdiff_emb[organization_idx] = organization_outputs[organization_idx].grad
                
                # Compute gradient-based reward for each organization
                for organization_idx in range(default_organization_num):
                    avg_contribution = contribution_per_organization[organization_idx]['average']
                    print("avg_contribution: ", avg_contribution)
                    grad_reward_eps_batch = self.grad_reward_epsilon(
                        -1 * grad_lossdiff_emb[organization_idx],
                        train_grad_noise,
                        avg_contribution,
                        train_loss.detach(),
                        epsilons[organization_idx],
                        sensitivity,
                        client_costs_ratio[organization_idx]
                    )
                    grad_reward_eps[organization_idx] += grad_reward_eps_batch
                
                # Average over all batches
                num_batches = len(batch_idxs_list)
                for organization_idx in range(default_organization_num):
                    grad_reward_eps[organization_idx] /= num_batches
                
                # Update epsilons for each org
                for organization_idx in range(default_organization_num):
                    step_update = step_size_for_epsilon_feedback * grad_reward_eps[organization_idx]
                    new_epsilon = epsilons[organization_idx] + step_update
                    if isinstance(new_epsilon, torch.Tensor):
                        new_epsilon = new_epsilon.detach().item()
                    # Clamp epsilon between [0.5, 1.0]
                    if step_size_for_epsilon_feedback > 0:
                        if new_epsilon < 0.5:
                            new_epsilon = 0.5
                        elif new_epsilon > 1:
                            new_epsilon = 1
                    epsilons[organization_idx] = new_epsilon
                
                epsilons_array.append(epsilons.copy())
                
                c_temp = []
                for key in range(default_organization_num):
                    if contribution_per_organization[key]['average'] > 0:
                        c_temp.append(contribution_per_organization[key]['average'])
                    else:
                        c_temp.append(0)
                
                cons = np.array([round(float(x), 3) for x in c_temp])
                # Combine cost ratio with contributions
                print("cons: ", cons)
                contribution_term = np.multiply(cons, np.array(client_costs_ratio[default_organization_num])**(1/5))
                privacy_term = sensitivity / np.array(epsilons[default_organization_num])
                rewards = (alpha * contribution_term) + (beta * privacy_term)
                
                contribution_term_array.append(contribution_term.tolist())
                privacy_term_array.append(privacy_term.tolist())
                rewards_array.append(rewards.tolist())
                
                # Distribute among these parties
                reward_distribution, t_r = self.distribute_rewards(rewards, total_tokens, organization_num)
                reward_distribution_array.append(reward_distribution.tolist())
                client_utility = reward_distribution - client_total_resources_for_training
                utility_array.append(client_utility.tolist())
                
                contributions_array.append([round(float(x), 3) for x in c_temp])
                
                print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(
                    i+1, train_loss.detach().numpy(), np.mean(train_auc_array_temp)))
                
                train_auc_array.append(np.mean(train_auc_array_temp))
                train_loss_np = train_loss.detach().numpy()
                train_loss_array.append(train_loss_np.item())

                if i == num_warmup_epochs:
                    utility = np.array(client_utility)
                    active_indices = [idx for idx, active in enumerate(self.active_clients) if active]
                    clients_with_negative_utility = np.where(utility <= -0.001)[0]
                    clients_with_negative_contribution = np.where(cons <= -0.0001)[0]
                    clients_to_remove = [active_indices[idx] for idx in clients_with_negative_utility]
                    clients_to_remove.extend([active_indices[idx] for idx in clients_with_negative_contribution])
                    for idx in clients_to_remove:
                        self.active_clients[idx] = False
                        print("Client ", idx, " removed in round ", i)
                if i >= num_warmup_epochs:
                    print("clients dropped: ", self.active_clients)

                # Validation
                if (i+1) % 1 == 0:
                    batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
                    val_auc_array_temp = []
                    val_loss_array_temp = []
                    
                    for batch_idxs in batch_idxs_list:
                        organization_outputs_for_val = {}
                        for organization_idx in range(default_organization_num):
                            organization_outputs_for_val[organization_idx] = organization_models[organization_idx](
                                X_val_vertical_FL[organization_idx][batch_idxs]
                            )
                        
                        organization_outputs_for_val_cat = None
                        for organization_idx in range(default_organization_num):
                            if organization_outputs_for_val_cat is None:
                                organization_outputs_for_val_cat = organization_outputs_for_val[organization_idx].float()
                            else:
                                organization_outputs_for_val_cat = torch.cat(
                                    (organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1
                                ).float()
                        
                        log_probs = top_model(organization_outputs_for_val_cat)
                        val_loss = criterion(log_probs, y_val[batch_idxs].type(torch.LongTensor))
                        predictions = torch.argmax(log_probs, dim=1)
                        correct = (predictions == y_val[batch_idxs]).sum().item()
                        total = y_val[batch_idxs].size(0)
                        val_auc = correct / total
                        
                        val_auc_array_temp.append(val_auc)
                        val_loss_array_temp.append(val_loss.detach().numpy())
                    
                    print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(
                        i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
                    val_auc_array.append(np.mean(val_auc_array_temp))
                    val_loss_array.append(np.mean(val_loss_array_temp))
            
                time_per_epoch_array.append(time.time() - st)

            # Testing phase
            organization_outputs_for_test = {}
            for organization_idx in range(default_organization_num):
                organization_outputs_for_test[organization_idx] = organization_models[organization_idx](
                    X_test_vertical_FL[organization_idx]
                )
            
            organization_outputs_for_test_cat = None
            for organization_idx in range(default_organization_num):
                if organization_outputs_for_test_cat is None:
                    organization_outputs_for_test_cat = organization_outputs_for_test[organization_idx].float()
                else:
                    organization_outputs_for_test_cat = torch.cat(
                        (organization_outputs_for_test_cat, organization_outputs_for_test[organization_idx]), 1
                    ).float()
            
            outputs = top_model(organization_outputs_for_test_cat)
            log_probs = outputs
            predictions = torch.argmax(log_probs, dim=1)
            correct = (predictions == y_test).sum().item()
            total = y_val.size(0)
            test_acc = correct / total
            print("contributions_array=", contributions_array)
            print("rewards_array=", rewards_array)
            print("epsilons_array", epsilons_array)
            
            # Return values
            return (train_loss_array, val_loss_array, train_auc_array, val_auc_array,
                    test_acc, epsilons_array, rewards_array, contributions_array, time_per_epoch_array)