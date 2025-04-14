import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append('../../torch_utils/')
from torch_model import torch_top_model_cifar10, torch_organization_model_cifar10
from feature_inference_attack import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === CONFIG ===
organization_num = 2  # change to 2, 3, 4, or 5 as required
organization_output_dim = [64] * organization_num
noise_dim = 100
top_hidden_units = [512, 256, 64]
top_output_dim = 10
num_epochs = 50
batch_size_new = 64

# === Load data ===
x_adv_all = torch.load('x_adv_all.pt').to(device)  # [N, 3, 32, width]
v_all = torch.load('v_all.pt').to(device)

visible_width = x_adv_all.shape[-1]
target_width = 32 - visible_width

# --- Split data into training and test sets ---
N = x_adv_all.size(0)
indices = torch.randperm(N)
train_size = int(0.8 * N)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]
x_adv_train = x_adv_all[train_idx]
v_train     = v_all[train_idx]
x_adv_test  = x_adv_all[test_idx]
v_test      = v_all[test_idx]

# === Load models ===
organization_models = {}
for org_idx in range(organization_num):
    model = torch_organization_model_cifar10(out_dim=organization_output_dim[org_idx])
    model.load_state_dict(torch.load(f"organization_model_{org_idx}.pt",
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    organization_models[org_idx] = model

top_model_input_dim = 64 * organization_num
top_model = torch_top_model_cifar10(top_model_input_dim, top_hidden_units, top_output_dim)
top_model.load_state_dict(torch.load("top_model.pt", map_location=device, weights_only=True))
top_model.to(device)
top_model.eval()

# === Generator ===
generator = Generator(noise_dim=noise_dim, target_width=target_width).to(device)
generator.train()  # train on training subset

# === Helper: Split inputs across organizations ===
def split_clients(x_known, x_generated, widths, device):
    synthetic_combined = torch.cat((x_known, x_generated), dim=3)
    client_inputs = []
    start_col = 0
    for i in range(len(widths)):
        w = widths[i]
        if i < 2:
            client_inputs.append(synthetic_combined[:, :, :, start_col:start_col+w])
        else:
            client_inputs.append(torch.zeros(x_known.size(0), 3, 32, w, device=device))
        start_col += w
    return client_inputs

# === Determine width splits ===
base_width = 32 // organization_num
remainder = 32 % organization_num
widths = [base_width + (1 if i < remainder else 0) for i in range(organization_num)]

# === Training Phase: Train generator on the training subset ===
optimizer_generator = optim.Adam(generator.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training Generator on training split:")
for epoch in range(num_epochs):
    generator.train()
    optimizer_generator.zero_grad()
    
    # Generate target features for the training set.
    r_t = torch.randn(x_adv_train.size(0), noise_dim).to(device)
    x_target_hat = generator(x_adv_train, r_t)  # shape: [train_size, 3, 32, target_width]
    
    # Split inputs: adversary known features and generated target patch.
    client_inputs = split_clients(x_adv_train, x_target_hat, widths, device)
    
    # Forward pass through each organization model and then through the top model.
    org_outputs = [organization_models[i](client_inputs[i]) for i in range(organization_num)]
    v_hat = top_model(torch.cat(org_outputs, dim=1))  # shape: [train_size, 10]
    
    loss = criterion(v_hat, v_train)
    loss.backward()
    optimizer_generator.step()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Training Logit MSE: {loss.item():.4f}")
    
print("Generator training complete.")

# === Inference Phase: Evaluate on held-out test set ===
generator.eval()
x_target_hat_test_batches = []
with torch.no_grad():
    for i in range(0, x_adv_test.size(0), batch_size_new):
        batch_x_adv = x_adv_test[i:i+batch_size_new]
        r_t = torch.randn(batch_x_adv.size(0), noise_dim).to(device)
        x_target_hat_test_batches.append(generator(batch_x_adv, r_t))
x_target_hat_test = torch.cat(x_target_hat_test_batches, dim=0)
print("Held-out test: Estimated target features shape:", x_target_hat_test.shape)

client_inputs_test = split_clients(x_adv_test, x_target_hat_test, widths, device)
org_outputs_test = []
with torch.no_grad():
    for i in range(organization_num):
        # Process each organization's data in batches to avoid OOM.
        client_input = client_inputs_test[i]
        out_batches = []
        for j in range(0, client_input.size(0), batch_size_new):
            batch = client_input[j:j+batch_size_new]
            out_batches.append(organization_models[i](batch))
        org_outputs_test.append(torch.cat(out_batches, dim=0))
v_hat_test = top_model(torch.cat(org_outputs_test, dim=1))
print("Held-out test: Predicted outputs shape:", v_hat_test.shape)

mse_test = F.mse_loss(v_hat_test, v_test)
print("Final held-out test logit-space MSE:", mse_test.item())