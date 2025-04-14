import torch
import torch.nn as nn
from dataset import * #get_dataset, partition_dataset, make_iter_loader_list, get_item_from_index, get_targets_from_index
from optimization import *#ZO_output_optim
from compressor import * #Scale_Compressor
from utils import * #init_models, init_optimizers, init_log_file, get_update_seq

from torch.autograd import Variable

import random
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, required=False, default=12341)
parser.add_argument('--framework_type', type=str, required=False, default="ZO", help='"ZOFO", "ZO", "FO"')
parser.add_argument('--dataset_name', type=str, required=False, default="CIFAR10", help='"CIFAR10" "MNIST"')
parser.add_argument('--model_type', type=str, required=False, default="SimpleResNet18", help='MLP, CNN, SimpleResNet18')
parser.add_argument('--n_party', type=int, required=False, default=3)
parser.add_argument('--client_output_size', type=int, required=False, default=10, help='MLP: 64. CNN 128. SimpleResNet18 10.')
parser.add_argument('--server_embedding_size', type=int, required=False, default=128, help="for MLP experiment only. ")
parser.add_argument('--client_lr', type=float, required=False, default=0.02)
parser.add_argument('--server_lr', type=float, required=False, default=0.02)
parser.add_argument('--batch_size', type=int, required=False, default=128, help="MNIST:64, CIFAR10:128")
parser.add_argument('--n_epoch', type=int, required=False, default=50, help="MNIST:100, CIFAR10:50")
parser.add_argument('--u_type', type=str, required=False, default="Uniform", help="Uniform, Normal, Coordinate")
parser.add_argument('--mu', type=float, required=False, default=0.001)
parser.add_argument('--d', type=float, required=False, default=1)
parser.add_argument('--sample_times', type=int, required=False, default=1, help="q")
parser.add_argument('--compression_type', type=str, required=False, default="None", help="None, Scale") ## remember to change to None. 
parser.add_argument('--compression_bit', type=int, required=False, default=8)
parser.add_argument('--response_compression_type', type=str, required=False, default="None", help="None, Scale") ## remember to change to None. 
parser.add_argument('--response_compression_bit', type=int, required=False, default=8)
parser.add_argument('--local_update_times', type=int, required=False, default=1, help="extra content")
parser.add_argument('--log_file_name', type=str, required=False, default="None")
parser.add_argument('--attack_type', type=str, required=False, default=None, help='"badvfl" for backdoor attack')
parser.add_argument('--severity', type=float, required=False, default=0.1, help='Attack severity (e.g., fraction of poisoned samples)')
args = parser.parse_args()

random_seed = args.random_seed

# framework
framework_type = args.framework_type # "ZOFO", "ZO", "FO"
dataset_name =  args.dataset_name # "CIFAR10" "MNIST"
# model
model_type =  args.model_type # MLP, CNN, SimpleResNet18, LinearResNet18
# model
n_party = args.n_party
n_client = n_party - 1
client_output_size = args.client_output_size #MLP: 64. CNN 128. SimpleResNet18 10.
server_embedding_size = args.server_embedding_size
# Training
server_lr = args.server_lr
client_lr = args.client_lr
batch_size = args.batch_size
n_epoch = args.n_epoch
u_type = args.u_type
mu = args.mu
d = args.d
sample_times = args.sample_times
# Special
compression_type = args.compression_type
compression_bit = args.compression_bit
response_compression_type = args.response_compression_type
response_compression_bit = args.response_compression_bit
# depreciate
local_update_times = args.local_update_times
# Log
log_file_name = args.log_file_name

# Set random seed
random.seed(random_seed)
torch.manual_seed(random_seed)

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Make dataset for server 0 and clients
trainset, testset = get_dataset(dataset_name)
print(1)
# Backdoor attack setup
if args.attack_type == "badvfl":
    print("Applying Backdoor Attack")
    poisoned_fraction = args.severity  # Fraction of data to poison
    target_label = 0  # Example target label for poisoned samples

    def insert_backdoor(dataset, fraction, target_label):
        dataset_list = list(dataset)  # Convert dataset to list for mutability
        total_samples = len(dataset_list)
        num_poisoned = int(total_samples * fraction)
        poisoned_indices = random.sample(range(total_samples), num_poisoned)

        for idx in poisoned_indices:
            data, label = dataset_list[idx]
            # Example backdoor modification: invert pixel values and assign target label
            data = 1 - data  # Invert pixel values
            label = target_label
            dataset_list[idx] = (data, label)

        return dataset_list  # Return the modified dataset as a list

    trainset = insert_backdoor(trainset, poisoned_fraction, target_label)

# Pass the poisoned dataset to partition_dataset (not as a DataLoader)
train_dataset_list, train_loader_list = partition_dataset(dataset_name, trainset, n_party, batch_size)
print(2)
train_dataset_list, train_loader_list = partition_dataset(dataset_name, trainset, n_party, batch_size)
train_iter_loader_list = make_iter_loader_list(train_loader_list, n_party)
n_training_batch_per_epoch_client = len(train_loader_list[0])
print(3)
# Loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn_coordinate = nn.CrossEntropyLoss(reduction="none")

# Zeroth order optimization
ZOO = ZO_output_optim(mu, u_type, client_output_size)
print(4)
# Compression setup
compressor = Scale_Compressor(bit=compression_bit)
response_compressor = Scale_Compressor(bit=response_compression_bit)

# Build the model list
models = init_models(model_type, n_party, train_dataset_list, device, client_output_size, server_embedding_size)

# Optimizers list
optimizers = init_optimizers(models, server_lr, client_lr)

# Record training data
field = ["epoch", "comm_round", "training_loss", "train_acc", "Bit"]
init_log_file(field, log_file_name)

# Generate the stimulated update sequence for clients
update_seq = get_update_seq(n_epoch, n_client, n_training_batch_per_epoch_client)
print(5)
print("len(update_seq): ", len(update_seq))
# Training loop (remains unchanged)
epoch = 0
batch = 0
running_loss = 0.0
correct = 0
raw_communication_size = 0

# Each iteration is one round
# print("update_seq: ", update_seq)
for m in update_seq:
    other_m_list = list(range(1, n_party))
    other_m_list.remove(m)

    # Forward pass for client
    try:
        inputs, labels, index = next(train_iter_loader_list[m])
    except StopIteration:
        train_iter_loader_list[m] = iter(train_loader_list[m])
        inputs, labels, index = next(train_iter_loader_list[m])

    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass for other clients
    inputs_list = [None] * n_party
    outputs_list = [None] * n_party
    compressed_list = [None] * n_party
    
    for all_m in range(1, n_party):
        inputs_list[all_m], _, _ = get_item_from_index(index, train_dataset_list[all_m])
        # print(f"Client {all_m} raw input shape: {inputs_list[all_m].shape}")
        # print(6)
        inputs_list[all_m] = inputs_list[all_m].to(device)
        # print(7)
        outputs_list[all_m] = models[all_m](inputs_list[all_m]).detach()
        # print(8)
        outputs_list[all_m] = compress_decompress(outputs_list[all_m], compression_type, compressor)
        # print(9)
    # Communication cost calculation (remains unchanged)
    
    # print(10)
    embedding_comm_cost_in_bit = compression_cost_in_bit(outputs_list[m], compression_type, compressor)
    raw_communication_size += embedding_comm_cost_in_bit

    # Framework-specific logic (remains unchanged)
    # ...

print('Finished Training')

# Testing
test_dataset_list, test_loader_list = partition_dataset(dataset_name, testset, n_party, batch_size)
correct = 0
total = 0

with torch.no_grad():
    for _, label, index in test_loader_list[0]:
        label = label.to(device)
        index = index.long()

        outputs_list = [None] * n_party
        for m in range(1, n_party):
            input_m, _, _ = get_item_from_index(index, test_dataset_list[m])
            input_m = input_m.to(device)
            outputs_list[m] = models[m](input_m)

        out_client = torch.cat(outputs_list[1:], dim=-1)
        outputs = models[0](out_client)

        _, predicted = torch.max(outputs.data, 1)
        total += label.shape[0]
        correct += (predicted == label).sum().item()

test_acc = correct / total
print(f'Accuracy of the network on the test set: {100 * test_acc:.2f}%')
append_log([0, 0, 0, test_acc, 0], log_file_name)
