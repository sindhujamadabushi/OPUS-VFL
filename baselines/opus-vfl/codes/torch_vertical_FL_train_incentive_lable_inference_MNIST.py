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
from torch.utils.data import DataLoader # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_organization_model, torch_top_model
import torch.nn.functional as F
from torch_generate_noise import add_Gaussian_noise
from set_environment_variables import set_environment
from label_inference_attack import MaliciousOptimizer, execute_label_inference_attack

import random
import argparse
import torch.nn.functional as F
import copy
import yaml
import argparse


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# default_organization_num = 5
one_hot = False


def load_config(config_path="../../configs/p2vfl.yaml"):
	with open(config_path, "r") as f:
		return yaml.safe_load(f)
	
config = load_config()
attack_type = 'label_inference'

data_type = 'original'                  
model_type = 'vertical'               
dname = config['experiment']['dataset']
num_iterations = config['experiment']['iterations']
# organization_num = 5
#incentive parameters
data_type = 'original'                 
model_type = "vertical"               
# default_organization_num = 6
delta = config['incentive']['delta']
sensitivity = config['incentive']['sensitivity']

alpha = 1   
beta = 1
dname = 'MNIST'
batch_size=config['experiment']['batch_size']
# step_size_for_epsilon_feedback = config['incentive']['step_size_for_epsilon_feedback'] 
step_size_for_epsilon_feedback = 0.1
# active_clients = [True] * (default_organization_num)
dp_during_train  =True
label_inference_malicious_client_idx = 2


class Vertical_FL_Train:
	
	''' Vertical Federated Learning Training Class'''
	def __init__(self, active_clients):
		if active_clients is None:
			self.active_clients = [True] * (6)
		else:
			self.active_clients = active_clients

	def grad_reward_epsilon(self, grad_lossdiff_emb, grad_noise, loss_diff, train_loss, epsilons, sensitivity, client_costs):
		# print("loss_diff: ", loss_diff)
		first_term = (torch.sum(grad_lossdiff_emb * grad_noise)* 100 * (1+loss_diff)/train_loss) * client_costs**(1/5)
		second_term = -sensitivity/epsilons**2
		return first_term + second_term
	
	def distribute_rewards(self, rewards, total_tokens, num_clients):
		total_rewards = np.ceil(
			rewards / np.sum(rewards)
			* (num_clients * (num_clients + 1) / 2)
		)
		total_tokens -= total_rewards
		return total_tokens, total_rewards

		
	def run(self, args, learning_rates, batch_size):

		''' Main function for the program'''
		data_type = args.data_type                  
		model_type = args.model_type                
		epochs = args.epochs 
		
		organization_num = args.organization_num
		total_tokens, client_costs_ratio, client_total_resources_for_training, num_warmup_epochs, epochs = set_environment(organization_num)
		default_organization_num = organization_num + 1
		    
		attribute_split_array = \
			np.zeros(organization_num+1).astype(int)  
		
		epsilons = [0.000001]#,0.001,0.001,0.001,0.001,0.001]
		print("organization_num: ", organization_num+1)
		for _ in range(1,int(organization_num)+1):
			random_number = random.uniform(0.5, 1)
			epsilons.append(random_number)
		print("epsilons initialization: ", epsilons)

		
		# dataset preprocessing
		if data_type == 'original':

			if args.dname == 'MNIST' or args.dname == 'FMNIST':
				# hidden_layer_size = trial.suggest_int('hidden_layer_size', 1, 100)
				file_path = "../../datasets/MNIST/{0}.csv".format(args.dname)
				X = pd.read_csv(file_path)
				y = X['class']
				# X = X[:10000]
				X = X.drop(['class'], axis=1)
				
				N, dim = X.shape
				columns = list(X.columns)
				
				attribute_split_array = \
					np.ones(len(attribute_split_array)).astype(int) * \
					int(dim/(organization_num+1))
				if np.sum(attribute_split_array) > dim:
					print('unknown error in attribute splitting!')
				elif np.sum(attribute_split_array) < dim:
					missing_attribute_num = (dim) - np.sum(attribute_split_array)
					attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
				else:
					print('Successful attribute split for multiple organizations')

	
		else:
			file_path = "./dataset/{0}.dat".format(args.dname)
			X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
	
		# initialize the arrays to be return to the main function
		train_loss_array = []
		val_loss_array = []
		val_auc_array = []
		train_auc_array = []
		
		if model_type == 'vertical':
			# define the attributes in each group for each organization
			attribute_groups = []
			attribute_start_idx = 0
			for organization_idx in range(organization_num+1):
				attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
				attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
				attribute_start_idx = attribute_end_idx
				# print('The attributes held by Organization {0}: {1}'.format(organization_idx, attribute_groups[organization_idx]))                        
			
			#attributes per organization
			for organization_idx in range(organization_num+1):
				print('The number of attributes held by Organization {0}: {1}'.format(organization_idx, len(attribute_groups[organization_idx])))
				
			# get the vertically split data with one-hot encoding for multiple organizations
			
			vertical_splitted_data = {}
			encoded_vertical_splitted_data = {}

			chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
			for organization_idx in range(organization_num+1):
				
				vertical_splitted_data[organization_idx] = \
					X[attribute_groups[organization_idx]].values#.astype('float32')
				
				
				encoded_vertical_splitted_data[organization_idx] = \
					chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
		
		
			if one_hot:
				chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
				for organization_idx in range(organization_num):
					
					vertical_splitted_data[organization_idx] = \
						X[attribute_groups[organization_idx]].values#.astype('float32')
					
					
					encoded_vertical_splitted_data[organization_idx] = \
						chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
			else:
				for organization_idx in range(organization_num+1):
					
					vertical_splitted_data[organization_idx] = \
						X[attribute_groups[organization_idx]].values#.astype('float32')
					
					encoded_vertical_splitted_data = vertical_splitted_data
			
					# encoded_vertical_splitted_data = self.feature_selection(vertical_splitted_data[organization_idx], 'pca')
			
			
			print('X shape:',X.shape)
			# set up the random seed for dataset split
			random_seed = 1001
			active_clients = self.active_clients
			
			# split the encoded data samples into training and test datasets
			X_train_vertical_FL = {}
			X_val_vertical_FL = {}
			X_test_vertical_FL = {}
			# selected_features = None
			
			for organization_idx in range(organization_num+1):
				test_set_size = 50  # or test_set_size = 1 if you want only one test sample

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
			for organization_idx in range(organization_num+1):
			
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
			#num_organization_hidden_layers = params['num_organization_hidden_layers']
			num_organization_hidden_units = 128   #params['num_organization_hidden_units']
			organization_hidden_units_array = [np.array([num_organization_hidden_units])]*(organization_num +1)  #* num_organization_hidden_layers
			organization_output_dim = np.array([64 for i in range(organization_num+1)])
			num_top_hidden_units = 64  #params['num_top_hidden_units']
			top_hidden_units = np.array([num_top_hidden_units])
			top_output_dim = 10
			
			# build the client models
			organization_models = {}
			for organization_idx in range(organization_num+1):
				organization_models[organization_idx] = \
					torch_organization_model(X_train_vertical_FL[organization_idx].shape[-1],\
									organization_hidden_units_array[organization_idx],
									organization_output_dim[organization_idx])
			# build the top model over the client models
			top_model = torch_top_model(sum(organization_output_dim), top_hidden_units, top_output_dim)
			# define the neural network optimizer
			optimizer = torch.optim.Adam(top_model.parameters(), lr=learning_rates[0], weight_decay=1e-5)  #params['learning_rate_top_model'] 2.21820943080931e-05, 0.00013203242287235933
			
			optimizer_organization_list = []
			for organization_idx in range(organization_num+1):
				optimizer_organization_list.append(torch.optim.Adam(organization_models[organization_idx].parameters(), lr=learning_rates[1], weight_decay=1e-5))    #params['learning_rate_organization_model']0.0004807528058809301,0.000100295059051174
				
			
			if attack_type == 'label_inference':
				optimizer_organization_list[label_inference_malicious_client_idx] = MaliciousOptimizer(
					organization_models[label_inference_malicious_client_idx].parameters(),
					lr=0.00001, 
					beta=0.9, 
					gamma=1.0, 
					r_min=config['attack']['r_min'],
					r_max=config['attack']['r_max']
				)	
			
			print('\nStart vertical FL......\n')   
			criterion = nn.CrossEntropyLoss()
			top_model.train()

			top_models = {i:copy.deepcopy(top_model) for i in range(1,default_organization_num)}
			weights_and_biases = {i:top_models[i].state_dict() for i in range(1,default_organization_num)}
			optimizer_individual_model = {i:torch.optim.Adam(top_models[i].parameters(), learning_rates[0]) for i in range(1,default_organization_num)}
			
			
			contributions_array =[]
			rewards_array = []
			epsilons_array = []
			utility_array = []
			contribution_term_array = []
			privacy_term_array = []
			reward_distribution_array = []

			top_model.train()
			print(client_costs_ratio)
			print(client_total_resources_for_training)
			epochs = 60
			for i in range(epochs):
				contribution_per_organization = {}
				# if i >= 30:   # drop out after ith epoch
				# 	self.active_clients = [False, False, False, False, False, True,False, True, True,False]    # specify which clients to be dropped out
				print('Epoch: ', i)
				st = time.time()
				batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
				grad_reward_eps = [0 for i in range(organization_num+1)]
					
				for bidx in range(len(batch_idxs_list)):
					grad_lossdiff_emb = {}
					batch_idxs = batch_idxs_list[bidx]
					train_auc_array_temp=[]
					optimizer.zero_grad()
					for organization_idx in range(organization_num+1):
						optimizer_organization_list[organization_idx].zero_grad()
					organization_outputs = {}
										
					for organization_idx in range(organization_num+1):
							organization_outputs[organization_idx] = \
								organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs])
							organization_outputs[organization_idx].retain_grad()
					
					# organization_outputs_cat = organization_outputs[0].float()
					organization_outputs_cat = torch.empty(0, device=organization_outputs[0].device)  # Ensure correct device placement
					 
					for organization_idx in range(organization_num+1):
							
							if dp_during_train:
								train_noise, train_grad_noise = add_Gaussian_noise(organization_outputs[organization_idx], epsilons[organization_idx], delta, sensitivity)
								organization_outputs_cat = torch.cat((organization_outputs_cat, train_noise), 1).float()
								# noise = train_noise - organization_outputs[organization_idx]
								# print("SNR for org: ",organization_idx, "in epoch ",i,  self.compute_snr(organization_outputs[organization_idx], noise))	
							else:
								organization_outputs_cat = torch.cat((organization_outputs_cat, organization_outputs[organization_idx]), 1).float()
						# else:	
						# 	if bidx == 0:
						# 		print("inactive client input ", organization_idx, " zeroed out")
						# 	zeroed_inputs = torch.zeros_like(organization_outputs[organization_idx])
						# 	organization_outputs_cat = torch.cat((organization_outputs_cat, zeroed_inputs), 1).float()
						
						
					organization_outputs_cat = organization_outputs_cat.float()  # Ensure it's a Float tensor
			
					outputs = top_model(organization_outputs_cat)

					log_probs = outputs
					train_loss = criterion(log_probs, y_train[batch_idxs])
					predictions = torch.argmax(log_probs, dim=1)
					correct = (predictions == y_train[batch_idxs]).sum().item()
					total = y_train[batch_idxs].size(0)
					train_auc = correct / total
					train_auc_array_temp.append(train_auc)

					train_loss.backward()  # backpropagate the loss
					
					inputs = {}
					
					for key in range(1, default_organization_num):
						temp = []
						for k in organization_outputs:
							if k == key:
								temp.append(torch.zeros_like(organization_outputs[k]))
							else:
								temp.append(organization_outputs[k])
						temp_tensor = torch.cat(temp, dim=1)
						inputs[key] = temp_tensor
					
					for key in range(1, default_organization_num):
						inputs[key] = inputs[key].detach()
						outputs_2 = top_models[key](inputs[key])

						train_loss_without = criterion(outputs_2, y_train[batch_idxs].type(torch.LongTensor))
						optimizer_individual_model[key].zero_grad()
					
						train_loss_without.backward()						
						optimizer_individual_model[key].step()
						weights_and_biases[key] = top_models[key].state_dict()

						if train_loss.item() != 0:
							net_contribution = ((train_loss_without.item() - train_loss.item())*100)/train_loss
						else:
							net_contribution = train_loss_without.item() * 100

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
					
					
					optimizer.step() # adjust parameters based on the calculated gradients 
					
					for organization_idx in range(organization_num):
						if attack_type == 'label_inference' and organization_idx == label_inference_malicious_client_idx:
							optimizer_organization_list[organization_idx].step_malicious()
						else:
							optimizer_organization_list[organization_idx].step()
					

				# train_loss_detached = train_loss.detach()
				for organization_idx in range(1, organization_num+1):
					# if active_clients[organization_idx]:
						grad_lossdiff_emb[organization_idx] = organization_outputs[organization_idx].grad	
					# else:
					# 	grad_lossdiff_emb[organization_idx] = 0
					
				for organization_idx in range(1, organization_num+1): 
					
					grad_reward_eps_batch = self.grad_reward_epsilon(
						-1 * grad_lossdiff_emb[organization_idx],
						train_grad_noise,
						contribution_per_organization[organization_idx]['average'],
						train_loss.detach(),
						epsilons[organization_idx], 
						sensitivity, 
						client_costs_ratio[organization_idx]
					)
					grad_reward_eps[organization_idx] += grad_reward_eps_batch
								
				num_batches = len(batch_idxs_list)
				for organization_idx in range(1, organization_num+1):
					grad_reward_eps[organization_idx] /= num_batches

				print("epsilons: ", epsilons)

				for organization_idx in range(1,organization_num+1):
					# print("grad_reward_eps[organization_idx]: ", grad_reward_eps[organization_idx])
					# if active_clients[organization_idx-1]: 
						step_update = step_size_for_epsilon_feedback * grad_reward_eps[organization_idx]
						new_epsilon = epsilons[organization_idx] + step_update
						if step_size_for_epsilon_feedback > 0:
							
							if new_epsilon < 0.5:
								new_epsilon = 0.5
							elif new_epsilon > 0.9999:
								new_epsilon = 0.9999
							epsilons[organization_idx] = new_epsilon
						# else:
							
						# 	epsilons = self.epsilons
					# else:
					# 	grad_reward_eps[organization_idx] = 0
				epsilons = [1.0]+[float(eps) for eps in epsilons[1:]]
				
				epsilons_array.append(epsilons.copy())

				c_temp = []
				c_temp_raw = []
				print("organization_num where there is error: ", organization_num)
				for key in range(1,organization_num+1):
					
					if contribution_per_organization[key]['average'] > 0:
						c_temp.append(contribution_per_organization[key]['average'])
					else:
						c_temp.append(0)
					c_temp_raw.append(contribution_per_organization[key]['average'])

				# print("raw contributions: ", c_temp_raw)
				cons = np.array([round(float(x), 3) for x in c_temp])
				print("contributions: ", len(cons))						
				# contributions_array.append([round(float(x), 3) for x in c_temp_raw])
				
				contribution_term = np.multiply(cons, np.array(client_costs_ratio[1:])**(1/5))
				privacy_term = sensitivity / np.array(epsilons[1:])
				rewards = (alpha*contribution_term) + (beta * privacy_term)
				
				contribution_term_array.append(contribution_term.tolist())
				privacy_term_array.append(privacy_term.tolist())
				rewards_array.append(rewards.tolist())
				
				print("contribution_term: ", contribution_term)
				print("privacy_term: ", privacy_term)

				reward_distribution, t_r = self.distribute_rewards(rewards, total_tokens, organization_num-1)
				reward_distribution_array.append(reward_distribution.tolist())
				# print("reward_distribution: ", reward_distribution)
				client_utility = reward_distribution - client_total_resources_for_training
				utility_array.append(client_utility.tolist())
				
				contributions_array.append([round(float(x), 3) for x in c_temp])
				
				if i == num_warmup_epochs:	
					# Remove clients with negative utility
					# contributions_array.append([round(float(x), 3) for x in c_temp])
				
					utility = np.array(client_utility)
					active_indices = [idx for idx, active in enumerate(active_clients) if active]
					clients_with_negative_utility = np.where(utility <= -0.001)[0]
					clients_with_negative_contribution = np.where(cons <= -0.0001)[0]
					clients_to_remove = [active_indices[idx] for idx in clients_with_negative_utility]
					clients_to_remove.extend([active_indices[idx] for idx in clients_with_negative_contribution])
					for idx in clients_to_remove:
						active_clients[idx] = False
						print("Client ", idx, " removed in round ", i)
				if i>= num_warmup_epochs:
					print("clients dropped: ", active_clients)

				print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, train_loss.detach().numpy(), np.mean(train_auc_array_temp)))
				train_auc_array.append(np.mean(train_auc_array_temp))
				train_loss=train_loss.detach().numpy()
				train_loss_array.append(train_loss.item())
				print("time taken for epoch: ", time.time()-st)
				if (i+1)%1 == 0:
					batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
				
					for batch_idxs in batch_idxs_list:
						val_auc_array_temp = []
						val_loss_array_temp = []
						organization_outputs_for_val = {}
						
						feature_mask_tensor_list = []
						for organization_idx in range(organization_num+1):
							organization_outputs_for_val[organization_idx] = organization_models[organization_idx](X_val_vertical_FL[organization_idx][batch_idxs])
							feature_mask_tensor_list.append(torch.full(organization_outputs_for_val[organization_idx].shape, organization_idx))
						organization_outputs_for_val_cat = organization_outputs_for_val[0].float()
					
						#DP
						if len(organization_outputs_for_val) >= 2:
							for organization_idx in range(1, organization_num+1):
								organization_outputs_for_val_cat = torch.cat((organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1).float()
			
						organization_outputs_for_val_cat = organization_outputs_for_val_cat.float()

						log_probs = top_model(organization_outputs_for_val_cat)
						val_loss = criterion(log_probs, y_val[batch_idxs].type(torch.LongTensor))
						predictions = torch.argmax(log_probs, dim=1)
						correct = (predictions == y_val[batch_idxs]).sum().item()
						total = y_val[batch_idxs].size(0)
						val_auc = correct / total
						val_auc_array_temp.append(val_auc)
						val_loss_array_temp.append(val_loss.detach().numpy())
				
					print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
					val_auc_array.append(np.mean(val_auc_array_temp))
					val_loss_array.append(np.mean(val_loss_array_temp))

			# testing
			organization_outputs_for_test = {}
			for organization_idx in range(organization_num+1):
				organization_outputs_for_test[organization_idx] = organization_models[organization_idx](X_test_vertical_FL[organization_idx])
			organization_outputs_for_test_cat = organization_outputs_for_test[0].float()
			for organization_idx in range(1, organization_num+1):
						organization_outputs_for_test_cat = torch.cat((organization_outputs_for_test_cat, organization_outputs_for_test[organization_idx]), 1).float()
			
					
			outputs = top_model(organization_outputs_for_test_cat)
			log_probs = outputs
			predictions = torch.argmax(log_probs, dim=1)
			
			correct = (predictions == y_test).sum().item()  
			total = y_val.size(0) 
			test_acc = correct / total 
		

			if attack_type == 'label_inference':
				malicious_embeddings_train = organization_models[label_inference_malicious_client_idx](X_train_vertical_FL[label_inference_malicious_client_idx])   # => shape (N, 98)
				malicious_embeddings_test = organization_models[label_inference_malicious_client_idx](X_test_vertical_FL[label_inference_malicious_client_idx])     # => shape (N_test, 98)
				label_inference_accuracy = execute_label_inference_attack(malicious_embeddings_train, malicious_embeddings_test, y_train, y_test)
				attack_accuracy = label_inference_accuracy

			return train_loss_array, val_loss_array, train_auc_array, val_auc_array,test_acc

			