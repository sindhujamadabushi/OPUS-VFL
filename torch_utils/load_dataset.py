import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST



def load_and_process_image_dataset(dname, organization_num, attribute_split_array):
    # hidden_layer_size = trial.suggest_int('hidden_layer_size', 1, 100)
    file_path = "../../datasets/{0}/{0}.csv".format(dname)
    X = pd.read_csv(file_path)
    y = X['class']
    # X = X[:100]
    X = X.drop(['class'], axis=1)
    print("X.shape: ", X.shape)
    N, dim = X.shape
    columns = list(X.columns)
    
    attribute_split_array = \
        np.ones(len(attribute_split_array)).astype(int) * \
        int(dim/organization_num)
    if np.sum(attribute_split_array) > dim:
        print('unknown error in attribute splitting!')
    elif np.sum(attribute_split_array) < dim:
        missing_attribute_num = (dim) - np.sum(attribute_split_array)
        attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
    else:
        print('Successful attribute split for multiple organizations')
    return columns, X, y, attribute_split_array

def load_and_process_tabular_dataset():
    pass

from torchvision.transforms import ToTensor

def load_cifar10(dataset_name, root='./data',subset_size=None):
    """
    Load CIFAR-10 dataset, concatenate train and test, and return images and labels.
    """
    if dataset_name == "CIFAR10":
        transform = ToTensor()
        train_set = CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = CIFAR10(root=root, train=False, download=True, transform=transform)
        
        # Combine train and test datasets
        train_loader = DataLoader(train_set, batch_size=len(train_set))
        test_loader = DataLoader(test_set, batch_size=len(test_set))
        
        train_images, train_labels = next(iter(train_loader))
        test_images, test_labels = next(iter(test_loader))
        
        # Concatenate train and test
        images = torch.cat((train_images, test_images), dim=0)
        labels = torch.cat((train_labels, test_labels), dim=0)

        if subset_size is not None:
            images = images[:subset_size]
            labels = labels[:subset_size]
        
        return images, labels
    else:
        raise ValueError("Only CIFAR-10 is supported for now.")

def vertical_split_images(images, k):
    # Convert the PyTorch tensor to a NumPy array.
    images_np = images.cpu().numpy()  # shape: (N, 3, 32, 32)
    
    N = images_np.shape[0]
    # Determine slice widths for a 32-pixel wide image
    base_width = 32 // k
    remainder = 32 % k
    widths = [base_width + (1 if i < remainder else 0) for i in range(k)]
    
    # Allocate empty arrays for each slice using NumPy.
    image_parts_np = [
        np.zeros((N, 3, 32, widths[i]), dtype=np.float32)
        for i in range(k)
    ]
    
    # For each image, split it into k vertical slices.
    for n in range(N):
        current_col = 0
        for i in range(k):
            end_col = current_col + widths[i]
            image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
            current_col = end_col
    
    # Print size of each split
    for idx, part in enumerate(image_parts_np):
        print(f"Split {idx + 1}: shape {part.shape}")  # Print shape of each split
    
    # Print total number of splits
    print(f"Total number of splits: {len(image_parts_np)}")
    
    # Return the list of NumPy arrays.
    return image_parts_np


