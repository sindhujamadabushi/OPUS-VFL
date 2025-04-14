import torch
from torch_model import torch_top_model
from torch_model import torch_organization_model

organization_num = 5
malicious_client_idx = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the top model with the same architecture
top_model = torch_top_model(input_dim=256, hidden_units=[64, 64], num_classes=10)
top_model.to(device)

# Recreate the organization models with input_dim=156 to match the saved checkpoint
organization_models = []
for organization_idx in range(organization_num):
    organization_model = torch_organization_model(input_dim=156, hidden_units=[128, 128], out_dim=64)
    organization_model.to(device)
    organization_models.append(organization_model)

# Load the saved weights for the organization models
for organization_idx in range(organization_num):
    try:
        state_dict = torch.load(f"organization_model_{organization_idx}.pth", map_location=device)
        organization_models[organization_idx].load_state_dict(state_dict, strict=False)  # Allow partial loading
        print(f"Organization model {organization_idx} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: organization_model_{organization_idx}.pth not found.")
    except RuntimeError as e:
        print(f"Error loading organization model {organization_idx}: {e}")

# Load the saved weights for the top model
try:
    state_dict = torch.load("top_model.pth", map_location=device)
    top_model.load_state_dict(state_dict, strict=False)  # Allow partial loading
    print("Top model loaded successfully.")
except FileNotFoundError:
    print("Error: top_model.pth not found.")
except RuntimeError as e:
    print(f"Error loading top model: {e}")

# Load test data
X_test_vertical_FL = torch.load("X_test_vertical_FL.pth")
y_test = torch.load("y_test.pth")

# Extract features from the malicious model
malicious_model = organization_models[malicious_client_idx]
malicious_model.eval()  # Set to evaluation mode
X_test = X_test_vertical_FL[malicious_client_idx].to(device)
malicious_features = malicious_model(X_test).detach()

# Define a simple classifier for reconstruction
classifier = torch.nn.Linear(malicious_features.size(1), 10)  # Adjust num_classes if needed
classifier.to(device)

# Prepare training setup for reconstruction
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
loss_reconstruction_array = []

# Train the classifier
num_epochs = 30000
for epoch in range(num_epochs):
    classifier.train()  # Set to training mode
    optimizer.zero_grad()

    # Make predictions and calculate loss
    predictions = classifier(malicious_features)
    loss_reconstruction = criterion(predictions, y_test.to(device))

    # Backpropagation and optimization
    loss_reconstruction.backward()
    optimizer.step()

    # Record the loss
    loss_reconstruction_array.append(loss_reconstruction.item())
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_reconstruction.item():.4f}")

# Evaluate reconstruction accuracy
classifier.eval()  # Set to evaluation mode
with torch.no_grad():
    test_predictions = classifier(malicious_features)
    reconstruction_accuracy = (test_predictions.argmax(dim=1) == y_test.to(device)).float().mean()
    print(f"Reconstruction Accuracy: {reconstruction_accuracy.item():.4f}")

