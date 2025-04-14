import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

class LabelInferenceHead(nn.Module):
        def __init__(self, in_features, out_features=10):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc4 = nn.Linear(128, out_features)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.dropout(self.relu(self.bn1(self.fc1(x))))
            x = self.dropout(self.relu(self.bn2(self.fc2(x))))
            x = self.dropout(self.relu(self.bn3(self.fc3(x))))
            x = self.fc4(x)
            return x


class MaliciousOptimizer(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4,
                 min_ratio=1.0, max_ratio=5.0):
        """
        Malicious SGD with adaptive gradient scaling (Fu et al., 2022)
        
        Args:
            gamma_lr_scale_up (float): Attack strength, used to adaptively scale gradients.
            min_grad_to_process (float): Minimum gradient threshold to be considered for scaling.
            min_ratio (float): Minimum ratio for scaling (prevents very small gradients).
            max_ratio (float): Maximum ratio for scaling (prevents explosion).
        """
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(MaliciousOptimizer, self).__init__(params, defaults)

        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.last_parameters_grads = []  # Store previous gradients

    def step_malicious(self, closure=None, near_minimum=False):
        """Performs a single optimization step with malicious gradient scaling."""
        loss = None
        if closure is not None:
            loss = closure()

        if len(self.last_parameters_grads) < len(self.param_groups):
            for _ in range(len(self.param_groups)):
                self.last_parameters_grads.append([])

        for id_group, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for id_parameter, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = grad.clone().detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, grad)
                    if nesterov:
                        grad = grad.add(momentum, buf)
                    else:
                        grad = buf

                # Skip malicious scaling if near minimum
                if not near_minimum:
                    # Initialize storage for previous gradients
                    if len(self.last_parameters_grads[id_group]) <= id_parameter:
                        self.last_parameters_grads[id_group].append(grad.clone().detach())
                    else:
                        last_grad = self.last_parameters_grads[id_group][id_parameter]
                        current_grad = grad.clone().detach()

                        # Avoid division by zero and very small gradients
                        scale_ratio = (current_grad / (last_grad + 1e-7)).clamp(min=self.min_ratio, max=self.max_ratio)
                        adaptive_scale = 1.0 + self.gamma_lr_scale_up * scale_ratio

                        # Apply scaling only if gradient is significant
                        if torch.norm(current_grad) > self.min_grad_to_process:
                            grad.mul_(adaptive_scale)

                        # Update stored gradient
                        self.last_parameters_grads[id_group][id_parameter] = current_grad

                # Gradient descent step
                p.data.add_(-group['lr'], grad)

        return loss
    
    def extract_bottom_model(client_model_path):
        model = torch.load(client_model_path)  # Load the attacker's bottom model
        return model
    


def execute_label_inference_attack(
    malicious_embeddings_train,
    malicious_embeddings_test,
    y_train,
    y_test,
    label_inference_model='label_inference_model_finetuned.pt'):

    model = LabelInferenceHead(in_features=128, out_features=10)

    label_inference_model = model.load_state_dict(torch.load('label_inference_model_finetuned.pt', map_location=device))

    
    optimizer = optim.Adam(label_inference_model.parameters(), lr=0.012*2)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(malicious_embeddings_train.cpu(), y_train.long())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_epochs = 1000
    for epoch in range(num_epochs):
        label_inference_model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = label_inference_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
        print(f"[Epoch {epoch+1}] Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")

    print(f"💾 Saving trained model to {save_model_path}")
    torch.save(label_inference_model.state_dict(), save_model_path)

    # Evaluation
    label_inference_model.eval()
    test_dataset = TensorDataset(malicious_embeddings_test.cpu(), y_test.long())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct_test, total_test = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = label_inference_model(batch_x)
            correct_test += (logits.argmax(1) == batch_y).sum().item()
            total_test += batch_y.size(0)
    print(f"Test Accuracy: {correct_test/total_test:.4f}")
    return correct_test / total_test

def extract_auxiliary_datapoints(X_train, y_train, num_aux_per_class=200):
    # Ensure labels are on CPU for indexing.
    y_cpu = y_train.cpu()
    unique_labels = torch.unique(y_cpu)
    aux_data_list = []
    aux_labels_list = []
    
    # Loop over each unique label and select a fixed number of samples.
    for lab in unique_labels:
        indices = (y_cpu == lab).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        # Randomly permute the indices and select num_aux_per_class samples.
        perm = torch.randperm(indices.size(0))
        selected = indices[perm][:num_aux_per_class]
        aux_data_list.append(X_train[selected])
        aux_labels_list.append(y_train[selected])
    
    if aux_data_list:
        aux_data = torch.cat(aux_data_list, dim=0)
        aux_labels = torch.cat(aux_labels_list, dim=0)
    else:
        aux_data = torch.Tensor([])
        aux_labels = torch.Tensor([])
    
    return aux_data, aux_labels

def generate_auxiliary_embeddings(malicious_model, aux_data, aux_labels, batch_size=256):
   
    malicious_model.eval()
    device = next(malicious_model.parameters()).device
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, aux_data.size(0), batch_size):
            batch = aux_data[i:i+batch_size].to(device)
            emb = malicious_model(batch)
            # If the model's output is not flat (e.g., for image data), flatten it.
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            embeddings_list.append(emb.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings, aux_labels

def fine_tune_label_inference_head(model, aux_embeddings, aux_labels, num_epochs=10, batch_size=16, lr=0.001):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create a dataset and loader from the auxiliary embeddings and labels.
    dataset = torch.utils.data.TensorDataset(aux_embeddings, aux_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / total
        accuracy = correct / total
        print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    return model

def extract_embeddings(model, X_data, batch_size=256):
  
    model.eval()
    device = next(model.parameters()).device
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, X_data.size(0), batch_size):
            batch = X_data[i:i+batch_size].to(device)
            emb = model(batch)
            # If the model output is multi-dimensional (e.g., [batch, C, H, W]), flatten it.
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            embeddings_list.append(emb.cpu())
    
    return torch.cat(embeddings_list, dim=0)


def infer_labels_from_embeddings(model, embeddings, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    
    with torch.no_grad():
        for i in range(0, embeddings.size(0), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())
    
    return torch.cat(predictions, dim=0)

