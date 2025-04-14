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
from torch.utils.data import DataLoader # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10
import numpy as np
import random
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
import yaml
import copy
import torch.optim as optim
from label_inference_attack import MaliciousOptimizer
from p2vfl_utils import split_tabular_data, concatenate_outputs, compute_contribution, update_epsilons, compute_rewards, load_or_save_data, load_or_save_vertical_split_data, batch_train_cifar10
from set_environment_variables import set_environment

one_hot = False
fs  = False
contribution_measurement = True
dp_during_train = True
dp_during_val = False
dp_during_test = True
utility_threshold = 0

label_inference_malicious_client_idx = 0
attack_type = 'label_inference'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

with open('/home/msindhuja/PERFACY/configs/p2vfl.yaml', 'r') as file:
    config = yaml.safe_load(file)
 

#VFL parameters
data_type = 'original'                  
model_type = 'vertical'               

delta = config['incentive']['delta']
sensitivity = config['incentive']['sensitivity']
alpha = config['incentive']['alpha']    #contr term
beta = config['incentive']['beta']  #priv term
# client_costs_ratio = config['incentive']['client_costs_ratio']
total_tokens = config['incentive']['total_tokens']
client_actual_resources = config['incentive']['client_actual_resources']

default_organization_num = 3
one_hot = False
class Vertical_FL_Train:
    ''' Vertical Federated Learning Training Class'''
    def __init__(self, active_clients):
        if active_clients is None:
            self.active_clients = [True] * (default_organization_num - 1)
        else:
            self.active_clients = active_clients

    #p2vfl
    def grad_reward_epsilon(self, grad_lossdiff_emb, grad_noise, loss_diff, train_loss, epsilons, sensitivity, client_costs, alpha, beta):
        train_loss = torch.tensor(train_loss)
        client_costs_tensor = torch.tensor(client_costs, device=device)
        
        grad_lossdiff_emb = grad_lossdiff_emb.to(device)
        loss_diff = loss_diff.to(device)
        train_loss = train_loss.to(device)
        
        first_term = alpha*((torch.sum(grad_lossdiff_emb * grad_noise)* 100 * (1+loss_diff)/train_loss) * client_costs_tensor**(1/5))
        second_term = beta*(-sensitivity/epsilons**2)
        return first_term + second_term
        
    def run(self, args, learning_rates, batch_size):

        ''' Main function for the program'''
        data_type = args.data_type                  
        epochs = args.epochs 
        organization_num = args.organization_num
        step_size_for_epsilon_feedback = args.step_size_for_epsilon_feedback
        active_clients = [True] * (organization_num+1)
          
        total_parties = organization_num+1
        total_tokens, client_costs_ratio, client_actual_resources, num_warmup_epochs = set_environment(total_parties, 'CIFAR10')
		
   
        attribute_split_array = \
            np.zeros(total_parties).astype(int)  
        
        #p2vfl
        epsilons = []   #,1,1,1,1,1]
        print("organization_num: ", total_parties)
        for _ in range(total_parties):
            random_number = random.uniform(0.5, 1)
            epsilons.append(random_number)
        print("epsilons initialization: ", epsilons)
        
        # dataset preprocessing
        if data_type == 'original':
            if args.dname == 'CIFAR10':
                
                transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
                print(f"Min pixel value: {img.min()}, Max pixel value: {img.max()}")
                
                test_set = CIFAR10(root='./data', train=False, download=True, transform=transfor_val)
                
                # Combine train and test datasets
                train_loader = DataLoader(train_set, len(train_set))
                test_loader = DataLoader(test_set, len(test_set))
                
                train_images, train_labels = next(iter(train_loader))
                test_images, test_labels = next(iter(test_loader))
                
                # Concatenate train and test
                X = torch.cat((train_images, test_images), dim=0)
                y = torch.cat((train_labels, test_labels), dim=0)

            
    
        else:
            file_path = "./dataset/{0}.dat".format(args.dname)
            X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
        # X = X[:30000]
        # y = y[:30000]
    
        # initialize the arrays to be return to the main function
        train_loss_array = []
        val_loss_array = []
        val_auc_array = []
        train_auc_array = []
        
        images_np = X.cpu().numpy()  # shape: (N, 3, 32, 32)
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

        print('X shape:',X.shape)
        # set up the random seed for dataset split
        random_seed = 1001
        
        # split the encoded data samples into training and test datasets
        X_train_vertical_FL = {}
        X_val_vertical_FL = {}
    

        for organization_idx in range(total_parties):
            if organization_idx == 0:
                # For the active party, use the labels.
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                y_train_val = y
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
                    train_test_split(X_train_val, y_train_val, test_size=0.20,random_state=random_seed)
                #, random_state=random_seed
            else:
                # For other parties, we don't have labels so use dummy labels.
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                dummy_labels = np.zeros(len(X_train_val))
                X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
                    train_test_split(X_train_val, dummy_labels, test_size=0.20,random_state=random_seed)
        
        
        train_loader_list, test_loader_list, val_loader_list = [], [], []
        for organization_idx in range(total_parties):
        
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
        organization_hidden_units_array = [np.array([num_organization_hidden_units])]*total_parties   #* num_organization_hidden_layers
        organization_output_dim = np.array([64 for i in range(total_parties)])
        num_top_hidden_units = 64  #params['num_top_hidden_units']
        # top_hidden_units = np.array([64,64])
        top_hidden_units = np.array([512,256,64])
        top_output_dim = 10
        
        organization_models = {}
                    
        for organization_idx in range(total_parties):
                    organization_models[organization_idx] = \
                        torch_organization_model_cifar10(out_dim=organization_output_dim[organization_idx]).to(device)
        
        print("organization_output_dim: ", organization_output_dim)
        top_model = torch_top_model_cifar10(sum(organization_output_dim), top_hidden_units, top_output_dim)
        top_model = top_model.to(device).float()
        

        print('\nStart vertical FL......\n')   
        criterion = nn.CrossEntropyLoss()
        top_model.train()


        top_models = {i:copy.deepcopy(top_model) for i in range(total_parties)}
        weights_and_biases = {i:top_models[i].state_dict() for i in range(total_parties)}
        # optimizer_individual_model = {i:torch.optim.Adam(top_models[i].parameters(), learning_rates[0]) for i in range(total_parties)}

        ######################################################################################################
            # DEFINE OPTIMIZERS AND SCHEDULERS #
        ######################################################################################################
        
        optimizer_bottom_models = {}
        scheduler_organization_list = []
        for organization_idx in range(total_parties):
            if organization_idx == label_inference_malicious_client_idx:
                optimizer = MaliciousOptimizer(
                    organization_models[label_inference_malicious_client_idx].parameters(),
                    lr=0.0005/2, #1, 0.5-0.05/6 #0.01-0.005
                    gamma_lr_scale_up=args.gamma_lr_scale_up,
                    min_ratio=1,
                    max_ratio=5
                )

                # Store in dictionary if you need to call malicious steps later
                optimizer_bottom_models[organization_idx] = optimizer
                scheduler_organization_list.append(None)
            else:
                optimizer = optim.SGD(organization_models[organization_idx].parameters(),
                                    lr=learning_rates[1],
                                    momentum=0.9,
                                    weight_decay=1e-3)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=args.g_bm)
                scheduler_organization_list.append(scheduler)
                optimizer_bottom_models[organization_idx] = optimizer

        # ---------------------------------------------------
        # For training: define optimizers & schedulers for universal top_model
        # ---------------------------------------------------
        
        top_model_optimizer = torch.optim.SGD(
                top_model.parameters(),
                lr=learning_rates[0],   # e.g., 0.1
                momentum=0.9,
                weight_decay=5e-4       # typical CIFAR-10 decay, or 1e-3, etc.
            )
        
        top_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            top_model_optimizer,
            milestones=[15,50, 75],
            gamma=args.g_tm
        )

        # ---------------------------------------------------
        # For contribution: define optimizers & schedulers for top_models
        # ---------------------------------------------------
        top_optimizer_individual_model = {
            organization_idx: torch.optim.SGD(
                top_models[organization_idx].parameters(),
                lr=learning_rates[0],   # e.g., 0.1
                momentum=0.9,
                weight_decay=5e-4       # typical CIFAR-10 decay, or 1e-3, etc.
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
            # END DEFINE OPTIMIZERS AND SCHEDULERS
        ######################################################################################################

        contributions_array =[]
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
            grad_reward_eps = [0 for i in range(total_parties)] #p2vfl
                  
            for bidx in range(len(batch_idxs_list)):
                grad_lossdiff_emb = {} 
                batch_idxs = batch_idxs_list[bidx]
                train_auc_array_temp=[]
                train_loss_array_temp = []

                top_model.zero_grad()
                for organization_idx in range(total_parties):
                    optimizer_bottom_models[organization_idx].zero_grad()       # zero grad for bottom model
                    
                organization_outputs = {}
                organization_outputs_cat, organization_outputs, train_grad_noise = concatenate_outputs(total_parties,organization_models,batch_idxs,X_train_vertical_FL,active_clients,dp_during_train,epsilons,delta,sensitivity,bidx)
                organization_outputs_cat = organization_outputs_cat.float()
                
                outputs = top_model(organization_outputs_cat)
                y_train = y_train.to(device)
                # print("y_train: ", y_train)
                log_probs = outputs
                train_loss = criterion(log_probs, y_train[batch_idxs])
                predictions = torch.argmax(log_probs, dim=1)
                correct = (predictions == y_train[batch_idxs]).sum().item()
                total = y_train[batch_idxs].size(0)
                    
                train_loss.backward()  # backpropagate the loss
                train_auc_array_temp.append(correct/total)
                train_loss_array_temp.append(train_loss.detach().cpu().item())

                torch.nn.utils.clip_grad_norm_(
                    top_model.parameters(),
                    max_norm=1.0
                )
                
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

                contribution_per_organization = compute_contribution(total_parties,inputs,top_model,criterion,y_train,batch_idxs,top_model_optimizer,weights_and_biases,train_loss,contribution_per_organization)
                
               
                top_model_optimizer.step()
                for organization_idx in range(total_parties):
                    if organization_idx == label_inference_malicious_client_idx:
                        
                        optimizer_bottom_models[organization_idx].step_malicious()
                        
                    else:
                        optimizer_bottom_models[organization_idx].step()        # bottom model
                                    
            top_model_scheduler.step()
            for sched in range(len(scheduler_organization_list)):
                if sched == label_inference_malicious_client_idx:
                    pass
                else:
                    scheduler_organization_list[sched].step()  # bottom model schedulers

            for sched in top_scheduler_individual_model:
                sched.step()

            train_loss_detached = train_loss.detach().cpu().numpy()
            
                
            # #####################################################################################################
            #         START grad_lossdiff_emb COMPUTATION
            # #####################################################################################################
            
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

            # #####################################################################################################
            #         END grad_lossdiff_emb COMPUTATION
            # #####################################################################################################
            
            
            # ------ACCUMULATE CONTRIBUTION TERMS------ #
            print("epsilons: ", epsilons)
            epsilons = update_epsilons(total_parties,active_clients,step_size_for_epsilon_feedback,grad_reward_eps,epsilons)
            c_temp = []
            for key in range(total_parties):
                c_temp.append(contribution_per_organization[key]['average'])
                
            cons = np.array([round(float(x), 3) for x in c_temp])
            contributions_array.append(c_temp)
            contribution_term, privacy_term, rewards, reward_distribution, client_utility, reward_distribution_array = compute_rewards(cons,client_costs_ratio,sensitivity,epsilons,alpha,beta,total_tokens,total_parties,reward_distribution_array,client_actual_resources)
            
            contribution_term_array.append(contribution_term)
            privacy_term_array.append(privacy_term)
            rewards_array.append(rewards)
            utility_array.append(client_utility)
            epsilons_array.append(epsilons.copy())
            # ------ACCUMULATE CONTRIBUTION TERMS------ #

            print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, np.mean(train_loss_array_temp),np.mean(train_auc_array_temp)))
            train_auc_array.append(np.mean(train_auc_array_temp))
            train_loss=train_loss.detach().cpu().numpy()
            train_loss_array.append(train_loss_array_temp[0])
                
            if (i+1)%1 == 0:
        
                top_model.eval()
                for org_model in organization_models.values():
                    org_model.eval()

                with torch.no_grad():

                    for organization_idx in range(total_parties):
                            X_val_vertical_FL[organization_idx] = X_val_vertical_FL[organization_idx].to(device)
                            # X_test_vertical_FL[organization_idx] = X_test_vertical_FL[organization_idx].to(device)
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
                    
                        #DP
                        if len(organization_outputs_for_val) >= 2:
                            for organization_idx in range(1, total_parties):
                                organization_outputs_for_val_cat = torch.cat((organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1).float()

                        
                        
                        organization_outputs_for_val_cat = organization_outputs_for_val_cat.float()

                        log_probs = top_model(organization_outputs_for_val_cat)
                        val_loss = criterion(log_probs, y_val[batch_idxs].to(device).long())
                        predictions = torch.argmax(log_probs, dim=1)
                        correct = (predictions == y_val[batch_idxs]).sum().item()
                        total = y_val[batch_idxs].size(0)
                        val_auc_array_temp.append(correct/total)
                        val_loss_array_temp.append(val_loss.detach().cpu().item())

                    print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1,np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
                    val_auc_array.append(np.mean(val_auc_array_temp))
                    val_loss_array.append(np.mean(val_loss_array_temp))

        if attack_type == 'label_inference':
            import os
            save_dir = "label_inference_data"
            os.makedirs(save_dir, exist_ok=True)

            # Save tensors
            torch.save(X_train_vertical_FL[label_inference_malicious_client_idx], os.path.join(save_dir, "X_train.pt"))
            torch.save(X_val_vertical_FL[label_inference_malicious_client_idx], os.path.join(save_dir, "X_val.pt"))
            torch.save(y_train, os.path.join(save_dir, "y_train.pt"))

            # Save model
            torch.save(organization_models[label_inference_malicious_client_idx].state_dict(), os.path.join(save_dir, "model_state_dict.pt"))
        
        
        
        
        time_taken = time.time()-st
        print("time_taken for one epoch= ", time_taken)
        print("train_auc_array =",train_auc_array)
        print("train_loss_array=", train_loss_array)
        print('val_auc_array=', val_auc_array)
        print("val_loss_array=",val_loss_array)
        
        print("learning_rates: ", learning_rates)
        return train_loss_array, val_loss_array, train_auc_array, val_auc_array, time_taken