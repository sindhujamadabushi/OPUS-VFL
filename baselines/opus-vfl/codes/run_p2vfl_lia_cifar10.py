import sys
sys.path.append('../../torch_utils/')
from torch_vertical_FL_train_incentive_label_inference_CIFAR10 import Vertical_FL_Train
import optuna
import argparse
import time
import sys
import random
import numpy as np
import torch


# MNIST Hyperparameters
hyper_parameters = {
	'learning_rate_top_model': 0.001/2, #/10, #1e-3, #0.000025
	'learning_rate_organization_model': 0.001/2, #5e-5/50, #0.00001
	'batch_size': 512 #2048
}
dataset = 'CIFAR10'
default_organization_num = 1  # put one less than the number of clients you like
num_default_epochs = 40
num_iterations = 5
dropout_probability = 0.7 # Probability of dropping a client
step_size_for_epsilon_feedback = 0.1
gamma_lr_scale_up = 0.7

one_hot = False
fs = False
g_tm = 0.525
g_bm = 0.525

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

	test_acc_array = []
	
	for i in range(num_iterations):
		vfl_model = Vertical_FL_Train(active_clients=active_clients)
		print("Iteration: ", i)
		train_loss_array, val_loss_array, train_auc_array, val_auc_array = vfl_model.run(args, learning_rates, batch_size)
		train_loss_array_sum += np.array(train_loss_array)
		val_loss_array_sum += np.array(val_loss_array)
		train_auc_array_sum += np.array(train_auc_array)
		val_auc_array_sum += np.array(val_auc_array)
		
	# Calculate averages
	train_loss_avg = (train_loss_array_sum / num_iterations).tolist()
	val_loss_avg = (val_loss_array_sum / num_iterations).tolist()
	train_auc_avg = (train_auc_array_sum / num_iterations).tolist()
	val_auc_avg = (val_auc_array_sum / num_iterations).tolist()
	
	
	print("train_auc_array = ", train_auc_avg)
	print("train_loss_array = ", train_loss_avg)
	print("val_auc_array = ", val_auc_avg)
	print("val_loss_array = ", val_loss_avg)
	

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
	parser.add_argument('--gamma_lr_scale_up', type=str, default=gamma_lr_scale_up, help='gamma_lr_scale_up')
	
	
	args = parser.parse_args()
	# learning_rates = [args.learning_rate_top, args.learning_rate_bottom]


	run_base_model(args)
