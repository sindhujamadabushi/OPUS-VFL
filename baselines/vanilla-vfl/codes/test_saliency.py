import numpy as np
import matplotlib.pyplot as plt

# Load the saliency maps from the file
saliency_maps = np.load('saliency_maps.npy')

# Print the shape to see if it matches your expectations (e.g., (N, H, W))
print("Saliency maps shape:", saliency_maps.shape)
print(saliency_maps[:5])
# Optionally, display one saliency map to visually confirm it
plt.figure(figsize=(5, 5))
plt.imshow(saliency_maps[0], cmap='hot')
plt.title("Sample Saliency Map")
plt.axis('off')
plt.show()