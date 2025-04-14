import sys
sys.path.append('../../torch_utils/')
from torch_vertical_FL_train_incentive_CIFAR100 import Vertical_FL_Train
import optuna
import argparse
import time
import sys
import random
import numpy as np
import torch


# MNIST Hyperparameters
hyper_parameters = {
	'learning_rate_top_model': 0.01, #/10, #1e-3, #0.000025
	'learning_rate_organization_model': 0.01, #5e-5/50, #0.00001
	'batch_size': 512 #2048
}
dataset = 'CIFAR100'
default_organization_num = 4  # put one less than the number of clients you like
num_default_epochs = 60
num_iterations = 10
dropout_probability = 0.7 # Probability of dropping a client
step_size_for_epsilon_feedback = 0.1

one_hot = False
fs = False
g_tm = 0.0525
g_bm = 0.0525

# Toggle this to make clients inactive
active_clients = [True]*int(default_organization_num)

batch_size = hyper_parameters['batch_size']
learning_rates = [hyper_parameters['learning_rate_top_model'], hyper_parameters['learning_rate_organization_model']]


def run_base_model(args):
	
	
	test_acc_sum = 0
	# Initialize arrays to accumulate results
	train_loss_array_sum = np.zeros(num_default_epochs)
	val_loss_array_sum = np.zeros(num_default_epochs)
	train_auc_array_sum = np.zeros(num_default_epochs)
	val_auc_array_sum = np.zeros(num_default_epochs)
	time_taken_sum = 0
	test_acc_array = []

	contribution_term_array_sum = np.zeros((args.epochs, default_organization_num))
	privacy_term_array_sum = np.zeros((args.epochs, default_organization_num))
	rewards_array_sum = np.zeros((args.epochs, default_organization_num))
	utility_array_sum = np.zeros((args.epochs, default_organization_num))
	epsilons_array_sum = np.zeros((args.epochs, default_organization_num+1))	

	
	for i in range(num_iterations):
		vfl_model = Vertical_FL_Train(active_clients=active_clients)
		print("Iteration: ", i)
		train_loss_array, val_loss_array, train_auc_array, val_auc_array, time_taken = vfl_model.run(args, learning_rates, batch_size)
		train_loss_array_sum += np.array(train_loss_array)
		val_loss_array_sum += np.array(val_loss_array)
		train_auc_array_sum += np.array(train_auc_array)
		val_auc_array_sum += np.array(val_auc_array)
		time_taken_sum += time_taken
		
	# Calculate averages
	train_loss_avg = (train_loss_array_sum / num_iterations).tolist()
	val_loss_avg = (val_loss_array_sum / num_iterations).tolist()
	train_auc_avg = (train_auc_array_sum / num_iterations).tolist()
	val_auc_avg = (val_auc_array_sum / num_iterations).tolist()
	time_taken_avg = time_taken_sum/num_iterations

	contribution_term_avg = (contribution_term_array_sum / num_iterations).tolist()
	privacy_term_avg = (privacy_term_array_sum / num_iterations).tolist()
	rewards_avg = (rewards_array_sum / num_iterations).tolist()
	utility_avg = (utility_array_sum / num_iterations).tolist()
	epsilons_avg = (epsilons_array_sum / num_iterations).tolist()
	
	# 	print("##### P2VFL CONTRIBUTION AND PRIVACY #####")
	print("contribution_term_array=",contribution_term_avg)
	print("privacy_term_array=",privacy_term_avg)
	print("rewards_array=",rewards_avg)
	print("utility_array=",utility_avg)
	print("epsilons_array=",epsilons_avg)
	

	return {
		'train_loss_avg': train_loss_avg,
		'val_loss_avg': val_loss_avg,
		'train_auc_avg': train_auc_avg,
		'val_auc_avg': val_auc_avg,
		
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='vertical FL')
	parser.add_argument('--dname', default=dataset, help='dataset name: AVAZU, ADULT')
	parser.add_argument('--epochs', type=int, default=num_default_epochs, help='number of training epochs')  
	parser.add_argument('--batch_type', type=str, default='mini-batch')
	parser.add_argument('--batch_size', type=int, default=batch_size)
	parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
	parser.add_argument('--model_type', default='vertical', help='define the learning methods: vertical or centralized')    
	parser.add_argument('--organization_num', type=int, default=default_organization_num, help='number of organizations, if we use vertical FL')
	parser.add_argument('--contribution_schem',  type=str, default='ig', help='define the contribution evaluation method')
	parser.add_argument('--attack', default='original', help='define the data attack or not')
	parser.add_argument('--deactivate_client', type=int, default=None, help='Index of client to deactivate (0-based)')
	parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')
	
	parser.add_argument('--learning_rate_top', type=float, default=learning_rates[0],
						help='Learning rate for the top model')
	parser.add_argument('--learning_rate_bottom', type=float, default=learning_rates[1],
						help='Learning rate for the organization models (bottom models)')

	parser.add_argument('--g_tm', type=float, default=g_tm,
						help='Learning rate for the top model')
	parser.add_argument('--g_bm', type=float, default=g_bm,
						help='Learning rate for the top model')
	parser.add_argument('--step_size_for_epsilon_feedback', type=str, default=step_size_for_epsilon_feedback, help='step_size_for_epsilon_feedback')
	
	
	args = parser.parse_args()
	# learning_rates = [args.learning_rate_top, args.learning_rate_bottom]


	run_base_model(args)

# import sys
# sys.path.append('../../torch_utils/')
# from baselines.p2vfl.codes.torch_vertical_FL_train_incentive_CIFAR10 import Vertical_FL_Train
# import optuna
# import argparse
# import time
# import sys
# import random
# import numpy as np
# import torch

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # MNIST Hyperparameters
# hyper_parameters = {
# 	'learning_rate_top_model': 1e-3/10, 
# 	'learning_rate_organization_model': 1e-2/6, #68
# 	'batch_size': 512
# }
# g_tm = 0.0125
# g_bm = 0.0125

# dataset = 'CIFAR10'
# num_default_epochs = 60
# num_iterations = 1
# step_size_for_epsilon_feedback = 0.1
# organization_num = 2


# active_clients = [True] * (organization_num+1)

# one_hot = False
# fs = False

# batch_size = hyper_parameters['batch_size']
# learning_rates = [hyper_parameters['learning_rate_top_model'], hyper_parameters['learning_rate_organization_model']]

# def run_base_model(args):
	
	
# 	test_acc_sum = 0
# 	# Initialize arrays to accumulate results
# 	train_loss_array_sum = np.zeros(args.epochs)
# 	val_loss_array_sum = np.zeros(args.epochs)
# 	train_auc_array_sum = np.zeros(args.epochs)
# 	val_auc_array_sum = np.zeros(args.epochs)
	
# 	contribution_term_array_sum = np.zeros((args.epochs, organization_num))
# 	privacy_term_array_sum = np.zeros((args.epochs, organization_num))
# 	rewards_array_sum = np.zeros((args.epochs, organization_num))
# 	utility_array_sum = np.zeros((args.epochs, organization_num))
# 	epsilons_array_sum = np.zeros((args.epochs, organization_num+1))	


# 	test_acc_array = []
	
# 	for i in range(num_iterations):
# 		vfl_model = Vertical_FL_Train()
# 		print("Iteration: ", i)
# 		train_loss_array, val_loss_array, train_auc_array, val_auc_array,contribution_term_array, privacy_term_array, rewards_array, utility_array, epsilons_array = vfl_model.run(args, learning_rates, batch_size)
# 		train_loss_array_sum += np.array(train_loss_array)
# 		val_loss_array_sum += np.array(val_loss_array)
# 		train_auc_array_sum += np.array(train_auc_array)
# 		val_auc_array_sum += np.array(val_auc_array)
		
# 		contribution_term_array_sum += np.array(contribution_term_array)
# 		privacy_term_array_sum += np.array(privacy_term_array)
# 		rewards_array_sum += np.array(rewards_array)
# 		utility_array_sum += np.array(utility_array)
# 		epsilons_array_sum += np.array(epsilons_array)	
	
# 	# Calculate averages
# 	train_loss_avg = (train_loss_array_sum / num_iterations).tolist()
# 	val_loss_avg = (val_loss_array_sum / num_iterations).tolist()
# 	train_auc_avg = (train_auc_array_sum / num_iterations).tolist()
# 	val_auc_avg = (val_auc_array_sum / num_iterations).tolist()
	
# 	contribution_term_avg = (contribution_term_array_sum / num_iterations).tolist()
# 	privacy_term_avg = (privacy_term_array_sum / num_iterations).tolist()
# 	rewards_avg = (rewards_array_sum / num_iterations).tolist()
# 	utility_avg = (utility_array_sum / num_iterations).tolist()
# 	epsilons_avg = (epsilons_array_sum / num_iterations).tolist()

# 	print("##### CREDENTIALS #####")
# 	print("dataset: ", dataset)
# 	print("organization_num: ", organization_num)
# 	print("hyper parameters: ", hyper_parameters)
# 	print("gamma scheduler top model: ", g_tm)
# 	print("gamma scheduler bottom model: ", g_bm)
# 	print("step_size_for_epsilon_feedback: ", step_size_for_epsilon_feedback)
	
# 	print("##### ACCURACY and LOSS #####")
# 	print("train_auc_array = ", train_auc_avg)
# 	print("train_loss_array = ", train_loss_avg)
# 	print("val_auc_array = ", val_auc_avg)
# 	print("val_loss_array = ", val_loss_avg)
	
# 	print("##### P2VFL CONTRIBUTION AND PRIVACY #####")
# 	print("contribution_term_array=",contribution_term_avg)
# 	print("privacy_term_array=",privacy_term_avg)
# 	print("rewards_array=",rewards_avg)
# 	print("utility_array=",utility_avg)
# 	print("epsilons_array=",epsilons_avg)

# 	return {
# 		'train_loss_avg': train_loss_avg,
# 		'val_loss_avg': val_loss_avg,
# 		'train_auc_avg': train_auc_avg,
# 		'val_auc_avg': val_auc_avg,
# 		'test_acc_array': test_acc_array
# 	}

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description='vertical FL')
# 	parser.add_argument('--dname', default=dataset, help='dataset name: AVAZU, ADULT')
# 	parser.add_argument('--epochs', type=int, default=num_default_epochs, help='number of training epochs')  
# 	parser.add_argument('--batch_type', type=str, default='mini-batch')
# 	parser.add_argument('--batch_size', type=int, default=batch_size)
# 	parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
# 	parser.add_argument('--model_type', default='vertical', help='define the learning methods: vertical or centralized')    
# 	parser.add_argument('--organization_num', default = organization_num,type=int, help='number of organizations, if we use vertical FL')
# 	parser.add_argument('--contribution_schem',  type=str, default='ig', help='define the contribution evaluation method')
# 	parser.add_argument('--attack', default='original', help='define the data attack or not')
# 	parser.add_argument('--deactivate_client', type=int, default=None, help='Index of client to deactivate (0-based)')
# 	parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')
# 	parser.add_argument('--step_size_for_epsilon_feedback', type=str, default=step_size_for_epsilon_feedback, help='step_size_for_epsilon_feedback')
	
# 	parser.add_argument('--g_tm', type=str, default=g_tm, help='step_size_for_epsilon_feedback')
# 	parser.add_argument('--g_bm', type=str, default=g_tm, help='step_size_for_epsilon_feedback')
	
# 	args = parser.parse_args()
	
# 	run_base_model(args)