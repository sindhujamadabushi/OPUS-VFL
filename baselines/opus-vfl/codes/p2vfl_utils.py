import torch
import sys
sys.path.append('../../torch_utils/')
from torch_generate_noise import add_Gaussian_noise
import numpy as np
import pickle
import os
import sys
sys.path.append('../../torch_utils/')
from load_dataset import load_and_process_image_dataset
from utils import load_dat, batch_split
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import time
import torch

def log_gpu_usage():
    """ Logs GPU usage using nvidia-smi. """
    os.system("nvidia-smi | tee -a gpu_log_2.txt")  # Append GPU log to a file

def monitor_gpu():
    """ Periodically logs GPU usage to check utilization. """
    for i in range(10):  # Log GPU usage 10 times during execution
        time.sleep(30)  # Wait 30 seconds before logging again
        log_gpu_usage()


def split_tabular_data(organization_num,columns,attribute_split_array,X):
    attribute_groups = []
    attribute_start_idx = 0
    for organization_idx in range(organization_num):
        attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
        attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
        attribute_start_idx = attribute_end_idx
    
    vertical_splitted_data = {}
    encoded_vertical_splitted_data = {}

    for organization_idx in range(organization_num):
        
        vertical_splitted_data[organization_idx] = \
            X[attribute_groups[organization_idx]].values#.astype('float32')
        
        encoded_vertical_splitted_data = vertical_splitted_data
    
    return encoded_vertical_splitted_data

def load_or_save_vertical_split_data(organization_num, dname, columns, attribute_split_array, X, folder="processed_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, f"vertical_splitted_data_organization_num_{organization_num}_dname_{dname}.pkl")
    
    if os.path.exists(filename):
        print("Loading encoded vertical split data from file...")
        with open(filename, 'rb') as file:
            encoded_vertical_splitted_data = pickle.load(file)
    else:
        print("Generating encoded vertical split data...")
        encoded_vertical_splitted_data = split_tabular_data(organization_num, columns, attribute_split_array, X)
        with open(filename, 'wb') as file:
            pickle.dump(encoded_vertical_splitted_data, file)
    
    return encoded_vertical_splitted_data

def load_or_save_data(dname, organization_num, attribute_split_array, folder="processed_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    x_filename = os.path.join(folder, f"X_clients_{organization_num}_dname_{dname}.pkl")
    y_filename = os.path.join(folder, f"y_clients_{organization_num}_dname_{dname}.pkl")
    columns_filename = os.path.join(folder, f"columns_clients_{organization_num}_dname_{dname}.pkl")
    attribute_filename = os.path.join(folder, f"attribute_split_array_clients_{organization_num}_dname_{dname}.pkl")
    
    if (os.path.exists(x_filename) and os.path.exists(y_filename) and 
        os.path.exists(columns_filename) and os.path.exists(attribute_filename)):
        
        with open(columns_filename, "rb") as f:
            columns = pickle.load(f)
        with open(x_filename, "rb") as f:
            X = pickle.load(f)
        with open(y_filename, "rb") as f:
            y = pickle.load(f)
        with open(attribute_filename, "rb") as f:
            attribute_split_array = pickle.load(f)
        
        print("Data loaded from saved files.")
    else:
        # Process data if file(s) do not exist
        columns, X, y, attribute_split_array = load_and_process_image_dataset(
            dname, organization_num, attribute_split_array
        )
        
        # Save the processed data
        with open(columns_filename, "wb") as f:
            pickle.dump(columns, f)
        with open(x_filename, "wb") as f:
            pickle.dump(X, f)
        with open(y_filename, "wb") as f:
            pickle.dump(y, f)
        with open(attribute_filename, "wb") as f:
            pickle.dump(attribute_split_array, f)
        
        print("Data processed and saved to files.")
    
    return columns, X, y, attribute_split_array


def load_or_save_vertical_split_data(organization_num, dname, columns, attribute_split_array, X, folder="processed_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, f"vertical_splitted_data_organization_num_{organization_num}_dname_{dname}.pkl")
    
    if os.path.exists(filename):
        print("Loading encoded vertical split data from file...")
        with open(filename, 'rb') as file:
            encoded_vertical_splitted_data = pickle.load(file)
    else:
        print("Generating encoded vertical split data...")
        encoded_vertical_splitted_data = split_tabular_data(organization_num, columns, attribute_split_array, X)
        with open(filename, 'wb') as file:
            pickle.dump(encoded_vertical_splitted_data, file)
    
    return encoded_vertical_splitted_data

def compute_contribution(organization_num,inputs,top_model, criterion,y_train,batch_idxs,top_optimizer_individual_model,weights_and_biases,train_loss,contribution_per_organization,check_key_1=False):
    st_compute_contribution = time.time()
    for key in range(organization_num):
        inputs[key] = inputs[key].detach()
        outputs_2 = top_model(inputs[key].float())

        train_loss_without = criterion(outputs_2, y_train[batch_idxs].to(device, dtype=torch.long))
        train_loss_without.backward(retain_graph=True)	

        #steps
        top_optimizer_individual_model.step()
        weights_and_biases[key] = top_model.state_dict()

        # print("train_loss_without: ", train_loss_without.item(), "train_loss", train_loss.item())
                
        if train_loss.item() != 0:
            net_contribution = ((train_loss_without.item() - train_loss.item())*100)/train_loss
        else:
            net_contribution = train_loss_without.item() * 100
        # print("net_contribution: ", net_contribution)
        if key in contribution_per_organization:
            contribution_per_organization[key]['sum'] += net_contribution.item()
            contribution_per_organization[key]['count'] += 1
            contribution_per_organization[key]['average'] = contribution_per_organization[key]['sum'] / contribution_per_organization[key]['count']
        else:
            contribution_per_organization[key] = {
                'sum': net_contribution,
                'count': 1,
                'average': net_contribution
            }
    # print("time taken to compute_contribution: ", time.time()-st_compute_contribution)
    return contribution_per_organization


















# def compute_contribution(organization_num,inputs,top_models,criterion,y_train,batch_idxs,optimizer_individual_model,weights_and_biases,train_loss,contribution_per_organization,check_key_1=False):
#     st_compute_contribution = time.time()
#     for key in range(organization_num):
#         inputs[key] = inputs[key].detach()
#         outputs_2 = top_models[key](inputs[key].float())

#         train_loss_without = criterion(outputs_2, y_train[batch_idxs].to(device, dtype=torch.long))
        
#         optimizer_individual_model[key].zero_grad()
        
#         train_loss_without.backward(retain_graph=True)	
        
#         train_loss_without.backward()
        
#         if check_key_1 and key == 1:
#             optimizer_individual_model[key].step_malicious()
#         else:
#             optimizer_individual_model[key].step()
        
#         weights_and_biases[key] = top_models[key].state_dict()
#         if train_loss.item() != 0:
#             net_contribution = ((train_loss_without.item() - train_loss.item())*100)/train_loss
#         else:
#             net_contribution = train_loss_without.item() * 100

#         if key in contribution_per_organization:
#             contribution_per_organization[key]['sum'] += net_contribution.item()
#             contribution_per_organization[key]['count'] += 1
#             contribution_per_organization[key]['average'] = contribution_per_organization[key]['sum'] / contribution_per_organization[key]['count']
#         else:
#             contribution_per_organization[key] = {
#                 'sum': net_contribution,
#                 'count': 1,
#                 'average': net_contribution
#             }
#     # print("time taken to compute_contribution: ", time.time()-st_compute_contribution)
#     return contribution_per_organization

def concatenate_outputs(organization_num,organization_models,batch_idxs,X_train_vertical_FL,active_clients,dp_during_train,epsilons,delta,sensitivity,bidx):
    st_concatenate_outputs = time.time()
    organization_outputs = {}
    
    # print("organization_models: ",organization_models)
    for organization_idx in range(organization_num):
            
            organization_outputs[organization_idx] = \
                organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs].to(device))
        
    for organization_idx in range(organization_num):
        organization_outputs[organization_idx].retain_grad()
    
    # Gaussian noise needs to be added here!!
    organization_outputs_cat = organization_outputs[0].float()
    for organization_idx in range(organization_num):
        if active_clients[organization_idx]:
            if dp_during_train:
                train_noise, train_grad_noise = add_Gaussian_noise(organization_outputs[organization_idx], epsilons[organization_idx], delta, sensitivity)
                if organization_idx != 0:
                    organization_outputs_cat = torch.cat((organization_outputs_cat, train_noise), 1).float()
                # noise = train_noise - organization_outputs[organization_idx]
                # print("SNR for org: ",organization_idx, "in epoch ",i,  self.compute_snr(organization_outputs[organization_idx], noise))	
            else:
                organization_outputs_cat = torch.cat((organization_outputs_cat, organization_outputs[organization_idx]), 1).float()
        else:	
            if bidx == 0:
                print("inactive client input ", organization_idx, " zeroed out")
            zeroed_inputs = torch.zeros_like(organization_outputs[organization_idx])
            organization_outputs_cat = torch.cat((organization_outputs_cat, zeroed_inputs), 1).float()
            
    organization_outputs_cat = organization_outputs_cat.float() 
    # print("time taken  to concatenate_outputs: ", time.time()-st_concatenate_outputs)
    return organization_outputs_cat, organization_outputs, train_grad_noise


def update_epsilons(organization_num,active_clients,step_size_for_epsilon_feedback,grad_reward_eps,epsilons):
    st_update_epsilons = time.time()
    for organization_idx in range(organization_num):
        # print("grad_reward_eps[organization_idx]: ", grad_reward_eps[organization_idx])
        if active_clients[organization_idx]: 
            # print("grad_reward_eps[organization_idx]: ", grad_reward_eps[organization_idx])
            step_update = step_size_for_epsilon_feedback * grad_reward_eps[organization_idx-1]
            new_epsilon = epsilons[organization_idx] + step_update
            if step_size_for_epsilon_feedback > 0:
                
                if new_epsilon < 0.5:
                    new_epsilon = 0.5
                elif new_epsilon > 1:
                    new_epsilon = 1
                epsilons[organization_idx] = new_epsilon
            # else:
                
            # 	epsilons = self.epsilons
        else:
            grad_reward_eps[organization_idx] = 0
    epsilons = [float(eps) for eps in epsilons]

    # print("time taken to update_epsilons: ", time.time()-st_update_epsilons)    
    return epsilons


def distribute_rewards(rewards, total_tokens, num_clients):
    total_rewards = np.ceil(
        rewards / np.sum(rewards)
        * (num_clients * (num_clients + 1) / 2)
    )
    total_tokens -= total_rewards
    return total_tokens, total_rewards

def compute_rewards(cons,client_costs_ratio,sensitivity,epsilons,alpha,beta,total_tokens,organization_num,reward_distribution_array,client_actual_resources ):
    st_compute_rewards = time.time()
   
    contribution_term = np.multiply(cons, np.array(client_costs_ratio[1:])**(1/5))
    privacy_term = sensitivity / np.array(epsilons)
    rewards = (alpha*contribution_term) + (beta * privacy_term)
    reward_distribution, t_r = distribute_rewards(rewards, total_tokens, organization_num-1)
    reward_distribution_array.append(reward_distribution.tolist())
    client_utility = reward_distribution - client_actual_resources
    # print("time taken to compute_rewards: ", time.time()-st_compute_rewards)
    return contribution_term, privacy_term, rewards, reward_distribution, client_utility, reward_distribution_array

def batch_train_cifar10(X_train_vertical_FL,batch_size, optimizer, organization_num,optimizer_organization_list, organization_models,dp_during_train, epsilons, delta, sensitivity,top_model, criterion, y_train, scaler):
    
    batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, 'mini-batch')
    train_auc_array_temp = []
    for batch_idxs in batch_idxs_list:
        # Zero all gradients
        optimizer.zero_grad()
        for org_idx in range(organization_num):
            optimizer_organization_list[org_idx].zero_grad()
            
        # Mixed precision context to reduce memory usage and improve performance
        with torch.amp.autocast('cuda'):  # Updated autocast usage
            organization_outputs = {}
            # Forward pass for each organization's model
            for org_idx in range(organization_num):
                organization_outputs[org_idx] = organization_models[org_idx](X_train_vertical_FL[org_idx][batch_idxs])
            
            # Concatenate outputs
            organization_outputs_cat = organization_outputs[0]
            for org_idx in range(1, organization_num):
                if dp_during_train:
                    train_noise, train_grad_noise = add_Gaussian_noise(
                        organization_outputs[org_idx], epsilons[org_idx], delta, sensitivity)
                    organization_outputs_cat = torch.cat((organization_outputs_cat, train_noise), dim=1)
                    # Optionally compute SNR here...
                else:
                    organization_outputs_cat = torch.cat((organization_outputs_cat, organization_outputs[org_idx]), dim=1)
                    
            outputs = top_model(organization_outputs_cat)
            train_loss = criterion(outputs, y_train[batch_idxs])
        
            # Compute training accuracy (AUC as defined here)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == y_train[batch_idxs]).sum().item()
            total = y_train[batch_idxs].size(0)
            train_auc = correct / total
            train_auc_array_temp.append(train_auc)
        
        # Backward pass with gradient scaling for mixed precision
        scaler.scale(train_loss).backward()
        
        # Step the optimizers
        scaler.step(optimizer)
        for org_idx in range(organization_num):
            scaler.step(optimizer_organization_list[org_idx])
        scaler.update()
        return train_auc_array_temp, train_loss

# import threading
# monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
# monitor_thread.start()