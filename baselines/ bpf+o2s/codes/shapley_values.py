import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

def objective_function(data, labels, feature_subset):
    # Subset the data to the selected features
    subset_data = data[:, feature_subset]
    # Train a decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(subset_data, labels)
    
    # Evaluate model performance
    predictions = model.predict(subset_data)
    return accuracy_score(labels, predictions)

def compute_shapley_values(data, labels, num_samples=100):
    
    num_features = data.shape[1]
    # num_features = 10
    shapley_values = np.zeros(num_features)

    for _ in range(num_samples):
        # Random permutation of features
        permuted_features = np.random.permutation(num_features)
        # Marginal contributions
        permuted_features
        subset = set()
        previous_value = 0
        st = time.time()
    
        for feature in permuted_features:
            subset.add(feature)
            current_value = objective_function(data, labels, list(subset))
            shapley_values[feature] += current_value - previous_value
            previous_value = current_value
        print("st: ", time.time()-st)

    # Normalize by the number of samples
    shapley_values /= num_samples
    print("shapley_values: ", shapley_values)
    np.save("shapley_values.npy", shapley_values)
    return shapley_values


