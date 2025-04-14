import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_auxiliary_dataset(csv_path, num_samples_per_class=400):
    df = pd.read_csv(csv_path)
    auxiliary_data = []
    auxiliary_labels = []
    train_indices = set()  
    for class_label in range(10):
        class_samples = df[df.iloc[:, 0] == class_label]
        class_samples = class_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        aux_samples = class_samples.iloc[:num_samples_per_class, 1:]  # Skip the first column (label)
        remaining_samples = class_samples.iloc[num_samples_per_class:].index
        train_indices.update(remaining_samples)
        auxiliary_data.append(aux_samples.values)
        auxiliary_labels.extend([class_label] * len(aux_samples))

    auxiliary_data = torch.tensor(np.concatenate(auxiliary_data, axis=0), dtype=torch.float32)
    auxiliary_labels = torch.tensor(auxiliary_labels, dtype=torch.long)
    return auxiliary_data, auxiliary_labels, train_indices


def label_inference(bottom_model, auxiliary_data, auxiliary_labels, ground_truth_labels, training_data, organization_num, malicious_client_idx):
    
    expected_input_dim = bottom_model.input_layer.in_features
    features_per_client = auxiliary_data.shape[1] // organization_num

    malicious_client_start = malicious_client_idx * features_per_client
    malicious_client_end = (malicious_client_idx + 1) * features_per_client

    auxiliary_data = auxiliary_data[:, malicious_client_start:malicious_client_end]

    print("auxiliary_data: ", auxiliary_data.shape)
    with torch.no_grad():
        auxiliary_embeddings = bottom_model(auxiliary_data)
    print("auxiliary_embeddings: ", auxiliary_embeddings.shape)
    surrogate_model = nn.Sequential(
        nn.Linear(auxiliary_embeddings.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, len(torch.unique(auxiliary_labels)))
    )
    optimizer = optim.Adam(surrogate_model.parameters(), lr=0.055, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    
    # Get class distributions
    aux_class_counts = Counter(auxiliary_labels.numpy())  # Convert tensor to numpy for counting
    train_class_counts = Counter(ground_truth_labels.numpy())

    # Print counts
    print("Auxiliary Dataset Class Distribution:", aux_class_counts)
    print("Training Dataset Class Distribution:", train_class_counts)

    
    for epoch in range(50):  
        optimizer.zero_grad()
        outputs = surrogate_model(auxiliary_embeddings)
        loss = criterion(outputs, auxiliary_labels)
        loss.backward()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.2f}')
        optimizer.step()
        
    training_embeddings = bottom_model(training_data[:ground_truth_labels.shape[0]])
    print("training_embeddings: ", training_embeddings.shape)
    
    with torch.no_grad():
        inferred_logits = surrogate_model(training_embeddings)
        inferred_labels = torch.argmax(inferred_logits, dim=1)
    
    accuracy = (inferred_labels == ground_truth_labels).float().mean().item()
    print(f'Inferred labels accuracy: {accuracy:.2f}')
    
    return inferred_labels

def compute_saliency_map(model, input_data, target_label):
    
    model.train()  

    input_data = input_data.to(device).float()  
    input_data.requires_grad = True 

    target_label = target_label.to(device).long() 

    
    if input_data.dim() == 3:  
        input_data = input_data.unsqueeze(0)

    # print(f"Final input_data shape: {input_data.shape}")  # Debugging

    output = model(input_data)  # Forward pass
    # print(f"Output requires_grad: {output.requires_grad}")  # Debugging

    loss = torch.nn.functional.cross_entropy(output, target_label.unsqueeze(0))  # Compute loss
    # print(f"Loss requires_grad: {loss.requires_grad}")  # Debugging

    if not loss.requires_grad:
        raise RuntimeError("Loss is not tracking gradients. Ensure input has requires_grad=True.")

    model.zero_grad() 
    loss.backward()  

    if input_data.grad is None:
        raise RuntimeError("Gradient not computed! Check requires_grad and input tracking.")

    saliency_map = input_data.grad.abs()  # Get saliency map

    if saliency_map.dim() == 4:  
        saliency_map = saliency_map.mean(dim=(0, 1))
    elif saliency_map.dim() == 3:  
        saliency_map = saliency_map.mean(dim=0)

    # print(f"Final saliency_map shape: {saliency_map.shape}")  # Debugging
    return saliency_map

def find_highest_saliency_region(saliency_map, window_size=5):
    num_features = saliency_map.shape[0]
    max_saliency = -1
    best_start = 0

    # Slide over the feature space
    for i in range(num_features - window_size + 1):
        avg_saliency = saliency_map[i : i + window_size].mean().item()
        if avg_saliency > max_saliency:
            max_saliency = avg_saliency
            best_start = i

    return best_start

def find_highest_saliency_region_2d(saliency_map, window_size=5):
    # saliency_map: tensor of shape [H, W]
    H, W = saliency_map.shape
    max_saliency = -1
    best_coords = (0, 0)
    for i in range(H - window_size + 1):
        for j in range(W - window_size + 1):
            patch = saliency_map[i:i+window_size, j:j+window_size]
            avg_saliency = patch.mean().item()
            if avg_saliency > max_saliency:
                max_saliency = avg_saliency
                best_coords = (i, j)
    return best_coords


def insert_saliency_trigger(data, labels, source_class, target_class, model, poisoning_budget=0.1, trigger_value=0.01, window_size=5):
    data = data.to(device)
    labels = labels.to(device)

    poisoned_data = data.clone()
    poisoned_labels = labels.clone()

    source_indices = (labels == source_class).nonzero(as_tuple=True)[0]
    num_poisoned = int(len(source_indices) * poisoning_budget)
    poisoned_indices = source_indices[torch.randperm(len(source_indices))[:num_poisoned]]

    for idx in poisoned_indices:
        sample = data[idx].unsqueeze(0).clone().to(device).requires_grad_(True)

        model.train()  
        sample_output = model(sample)  
        sample_output.requires_grad_(True)  

        target_label = torch.tensor(target_class, device=device)

        # Debug check
        # print(f"Sample requires_grad: {sample.requires_grad}, grad_fn: {sample.grad_fn}")
        # print(f"Sample_output requires_grad: {sample_output.requires_grad}, grad_fn: {sample_output.grad_fn}")

        saliency_map = compute_saliency_map(model, sample, target_label)

        if saliency_map.dim() == 2:  # Expected [H, W]
            spatial_saliency = saliency_map
        elif saliency_map.dim() == 3:  # Unexpected [C, H, W], reduce over C
            spatial_saliency = saliency_map.mean(dim=0)
        else:
            raise ValueError(f"Unexpected saliency_map shape: {saliency_map.shape}")

        best_i, best_j = find_highest_saliency_region_2d(spatial_saliency, window_size)

        # Insert trigger patch
        H, W = spatial_saliency.shape
        for c in range(sample.shape[1]):  # Assuming (C, H, W)
            for di in range(window_size):
                for dj in range(window_size):
                    if best_i + di < H and best_j + dj < W:
                        poisoned_data[idx, c, best_i+di, best_j+dj] += trigger_value
        poisoned_labels[idx] = target_class

    return poisoned_data, poisoned_labels


def compute_asr(top_model, X_test_vertical_FL, y_test, source_class, target_class, organization_models, organization_num):
    """ Computes the Attack Success Rate (ASR) """
    
    # print("Source class count:", (y_test == source_class).sum().item())
    # Get poisoned test samples (originally from `source_class`)
    poisoned_idxs = (y_test == source_class).nonzero(as_tuple=True)[0]

    # Forward pass poisoned samples
    organization_outputs_for_test = {}
    for org_idx in range(organization_num):
        organization_outputs_for_test[org_idx] = organization_models[org_idx](X_test_vertical_FL[org_idx][poisoned_idxs])

    # Aggregate outputs
    organization_outputs_for_test_cat = torch.cat([organization_outputs_for_test[org_idx] for org_idx in range(organization_num)], dim=1).float()
    
    # Get top model predictions
    log_probs = top_model(organization_outputs_for_test_cat)
    poisoned_predictions = torch.argmax(log_probs, dim=1)

    # Compute ASR: how many poisoned samples were misclassified as `target_class`
    asr = (poisoned_predictions == target_class).float().mean().item()
    
    # print(f" Attack Success Rate (ASR): {asr:.4f}")
    return asr

def flip_labels(y_train, flip_ratio=0.3, num_classes=10):
    """
    Flip a certain percentage of labels in y_train.

    Args:
        y_train (torch.Tensor): Tensor of labels.
        flip_ratio (float): Fraction of labels to flip (default: 0.35 for 35% flip).
        num_classes (int): Total number of classes (default: 10 for classification).

    Returns:
        torch.Tensor: Modified labels with 65% correct and 35% randomly flipped.
    """
    y_train = y_train.clone()  # Avoid modifying original tensor
    num_samples = len(y_train)
    num_to_flip = int(num_samples * flip_ratio)  # 35% of samples

    # Get random indices to flip
    flip_indices = torch.randperm(num_samples)[:num_to_flip]

    for idx in flip_indices:
        original_label = y_train[idx].item()
        new_label = original_label
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)  # Pick a different class
        y_train[idx] = new_label  # Assign new flipped label

    return y_train

# Check how many were flipped
# num_flipped = (y_train != flipped_y_train).sum().item()
# print(f"Total samples: {len(y_train)}, Flipped samples: {num_flipped}")
