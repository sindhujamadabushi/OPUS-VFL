import numpy as np
import matplotlib.pyplot as plt
import torch

def hermite_extrapolation(client_data, labels, num_known_points=10, interp_points=50):
    """
    Perform Hermite extrapolation and assign labels to extrapolated points.

    Args:
        client_data (np.ndarray): Raw data of shape (num_samples, num_features).
        labels (np.ndarray): Labels corresponding to the client data.
        num_known_points (int): Number of anchor points for interpolation.
        interp_points (int): Number of extrapolated points.

    Returns:
        synthetic_data (np.ndarray): Extrapolated data of shape (interp_points, num_features).
        synthetic_labels (np.ndarray): Labels for extrapolated data of shape (interp_points,).
    """
    num_samples, num_features = client_data.shape
    synthetic_data = []

    # Select known points (indices in the raw dataset)
    known_indices = np.linspace(0, num_samples - 1, num_known_points).astype(int)
    print("Known indices in the training data:", known_indices)
    
    # Extract labels for the known indices
    known_labels = labels[known_indices]
    print("Labels for known indices:", known_labels)
    known_labels = known_labels.to_numpy()  # Add this line

    print("Labels for known indices:", known_labels)


    # Compute mapping of extrapolated points to known points
    x_interp = np.linspace(0, num_samples - 1, interp_points)
    mapping = np.digitize(x_interp, known_indices, right=True) - 1
    mapping = np.clip(mapping, 0, num_known_points - 1)

    # Assign labels to extrapolated points
    synthetic_labels = known_labels[mapping]

    # Iterate through each feature
    for feature_index in range(num_features):
        feature_values = client_data[:, feature_index]

        # Extract values at known points
        f_points = feature_values[known_indices]

        # Compute numerical derivatives
        f_derivatives = np.gradient(f_points, known_indices)

        # Perform interpolation for the feature
        feature_synthetic = hermite_interpolation(x_interp, known_indices, f_points, f_derivatives)
        synthetic_data.append(feature_synthetic)

    return np.array(synthetic_data).T, synthetic_labels  # Return data and labels


def hermite_interpolation(x, x_points, f_points, f_derivatives):
    """
    Perform piecewise cubic Hermite interpolation for a single feature.
    """
    interpolated_values = np.zeros_like(x)
    for i in range(len(x_points) - 1):
        x_i = x_points[i]
        x_i1 = x_points[i + 1]
        h = x_i1 - x_i

        # Basis functions
        h1 = lambda x: (1 + 2 * (x - x_i) / h) * ((x - x_i1) / h) ** 2
        h2 = lambda x: (1 + 2 * (x - x_i1) / h) * ((x - x_i) / h) ** 2
        h3 = lambda x: (x - x_i) * ((x - x_i1) / h) ** 2
        h4 = lambda x: (x - x_i1) * ((x - x_i) / h) ** 2

        # Indices for the current segment
        segment_mask = (x >= x_i) & (x <= x_i1)
        x_segment = x[segment_mask]

        # Interpolation for the segment
        interpolated_values[segment_mask] = (
            h1(x_segment) * f_points[i]
            + h2(x_segment) * f_points[i + 1]
            + h3(x_segment) * f_derivatives[i] * h
            + h4(x_segment) * f_derivatives[i + 1] * h
        )
    return interpolated_values

