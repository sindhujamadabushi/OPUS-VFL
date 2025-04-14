import torch
import sys
sys.path.append('../../torch_utils/')
from label_inference_attack import LabelInferenceHead, extract_auxiliary_datapoints, generate_auxiliary_embeddings, fine_tune_label_inference_head, extract_embeddings
import os
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_label_inference_accuracy(predicted_labels, true_labels):
    # Ensure true_labels are on CPU
    true_labels = true_labels.cpu()
    correct = (predicted_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def compute_topk_accuracy(outputs, targets, k=5):
    # Make sure they're on the same device (GPU)
    targets = targets.to(outputs.device)

    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    topk_correct = correct.any(dim=1).float().sum()
    acc = topk_correct.item() / targets.size(0)
    return acc

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

# for top-k
def infer_labels_from_embeddings2(model, embeddings, batch_size=256, k=4):
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []

    with torch.no_grad():
        for i in range(0, embeddings.size(0), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            outputs = model(batch)
            all_outputs.append(outputs.cpu())  # keep them on CPU

    # Now combine them into one tensor: shape [total_samples, num_classes]
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs

# Load saved debug data
save_dir = "label_inference_data"
X_train_loaded = torch.load(os.path.join(save_dir, "X_train.pt"))
X_val_loaded = torch.load(os.path.join(save_dir, "X_val.pt"))
y_train_loaded = torch.load(os.path.join(save_dir, "y_train.pt"))

malicious_model = torch_organization_model_cifar10(out_dim=64).to(device)

# Load model (make sure to initialize the same model architecture first)
malicious_model.load_state_dict(
    torch.load(os.path.join(save_dir, "model_state_dict.pt"))
)

aux_data, aux_labels = extract_auxiliary_datapoints(
X_train = X_train_loaded,
y_train = y_train_loaded,
num_aux_per_class=3000  # Adjust as needed (e.g., for CIFAR-10, 4 per class gives ~40 samples)
)

aux_embeddings, aux_labels = generate_auxiliary_embeddings(malicious_model, aux_data, aux_labels, batch_size=256)

in_features = aux_embeddings.shape[1]

# Instantiate the label inference head.
label_inference_model = LabelInferenceHead(in_features=in_features, out_features=10)

# Fine-tune the model with the auxiliary data.
fine_tuned_model = fine_tune_label_inference_head(
    model=label_inference_model,
    aux_embeddings=aux_embeddings,
    aux_labels=aux_labels,
    num_epochs=230,  #org_num1-400
    batch_size=128,
    lr=0.01
)

# Extract embeddings for training and validation data.
malicious_embeddings_train = extract_embeddings(malicious_model, X_train_loaded, batch_size=256)
malicious_embeddings_test  = extract_embeddings(malicious_model, X_val_loaded, batch_size=256)

label_inference_model = fine_tuned_model
# label_inference_model = LabelInferenceHead(in_features=malicious_embeddings_train.shape[1], out_features=10)
# label_inference_model.to(device)

# state_dict = torch.load('label_inference_model_finetuned.pt', map_location=device)
# label_inference_model.load_state_dict(state_dict)
# label_inference_model.eval()

predicted_labels = infer_labels_from_embeddings(label_inference_model, malicious_embeddings_train, batch_size=256)
train_outputs = infer_labels_from_embeddings2(label_inference_model, malicious_embeddings_train, k=5)

accuracy = compute_label_inference_accuracy(predicted_labels, y_train_loaded)
top_k_accuracy = compute_topk_accuracy(train_outputs,y_train_loaded, k=5 )
print("Label Inference Accuracy top 5:", top_k_accuracy)
print("Label Inference Accuracy top 1:", accuracy)