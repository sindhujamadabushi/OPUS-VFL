import sys
sys.path.append('../../torch_utils/')
from torch_vertical_FL_train_incentive_MNIST import Vertical_FL_Train
import optuna
import argparse
import time
import sys
import random
import numpy as np
import torch

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# MNIST Hyperparameters
hyper_parameters = {
    'learning_rate_top_model': 0.001/86,
    'learning_rate_organization_model': 0.0005/86,
    'batch_size': 512
}
dataset = 'MNIST'
default_organization_num = '5'  # We use 25 orgs exactly
num_default_epochs = 60
num_iterations = 1

one_hot = False
fs = False

batch_size = hyper_parameters['batch_size']
learning_rates = [
    hyper_parameters['learning_rate_top_model'],
    hyper_parameters['learning_rate_organization_model']
]

def run_base_model(args):
    test_acc_sum = 0
    train_loss_array_sum = np.zeros(args.epochs)
    val_loss_array_sum = np.zeros(args.epochs)
    train_auc_array_sum = np.zeros(args.epochs)
    val_auc_array_sum = np.zeros(args.epochs)

    test_acc_array = []

    for i in range(num_iterations):
        
        epsilons_array_all = []
        rewards_array_all = []
        contributions_array_all = []
        epoch_times_all = []


        vfl_model = Vertical_FL_Train(active_clients=active_clients)
        print("Iteration: ", i)
        (train_loss_array, val_loss_array, train_auc_array, val_auc_array, test_acc,
        epsilons_array, rewards_array, contributions_array, time_per_epoch_array) = \
        vfl_model.run(args, learning_rates, batch_size)
        
        train_loss_array_sum += np.array(train_loss_array)
        val_loss_array_sum   += np.array(val_loss_array)
        train_auc_array_sum  += np.array(train_auc_array)
        val_auc_array_sum    += np.array(val_auc_array)
        test_acc_array.append(test_acc)
        test_acc_sum += test_acc

        epsilons_array_all.append(np.array(epsilons_array))        # shape: (epochs, num_orgs)
        rewards_array_all.append(np.array(rewards_array))          # shape: (epochs, num_orgs)
        contributions_array_all.append(np.array(contributions_array))  # shape: (epochs, num_orgs)
        epoch_times_all.append(np.array(time_per_epoch_array))     # shape: (epochs,)

    train_loss_avg = (train_loss_array_sum / num_iterations).tolist()
    val_loss_avg   = (val_loss_array_sum   / num_iterations).tolist()
    train_auc_avg  = (train_auc_array_sum  / num_iterations).tolist()
    val_auc_avg    = (val_auc_array_sum    / num_iterations).tolist()
    test_acc_avg   = test_acc_sum / num_iterations

    epsilons_avg = np.mean(epsilons_array_all, axis=0).tolist()
    rewards_avg = np.mean(rewards_array_all, axis=0).tolist()
    contributions_avg = np.mean(contributions_array_all, axis=0).tolist()
    epoch_times_avg = np.mean(epoch_times_all, axis=0).tolist()

    print("train_auc_array = ", train_auc_array)
    print("train_loss_array = ", train_loss_array)
    print("val_auc_array = ", val_auc_array)
    print("val_loss_array = ", val_loss_array)
    print("test_acc = ", test_acc_avg)


    print("epsilon_array=", epsilons_avg)
    print("rewards_array=", rewards_avg)
    print("contributions_array=", contributions_avg)
    print("Average time per epoch:", epoch_times_avg)

    return {
        'train_loss_avg': train_loss_avg,
        'val_loss_avg': val_loss_avg,
        'train_auc_avg': train_auc_avg,
        'val_auc_avg': val_auc_avg,
        'test_acc_avg': test_acc_avg,
        'test_acc_array': test_acc_array
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertical FL')
    parser.add_argument('--dname', default=dataset, help='dataset name: AVAZU, ADULT')
    parser.add_argument('--epochs', type=int, default=num_default_epochs, help='number of training epochs')
    parser.add_argument('--batch_type', type=str, default='mini-batch')
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
    parser.add_argument('--model_type', default='vertical', help='define the learning methods: vertical or centralized')
    parser.add_argument('--organization_num', default=default_organization_num, type=int,
                        help='number of organizations, if we use vertical FL')
    parser.add_argument('--contribution_schem',  type=str, default='ig', help='define the contribution evaluation method')
    parser.add_argument('--attack', default='original', help='define the data attack or not')
    parser.add_argument('--deactivate_client', type=int, default=None, help='Index of client to deactivate (0-based)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')

    args = parser.parse_args()
    # Keep the default_organization_num assignment
    default_organization_num = args.organization_num + 1

    # Use 25 active clients
    active_clients = [True] * args.organization_num

    run_base_model(args)