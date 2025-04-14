from sklearn.ensemble import IsolationForest
import torch
import numpy as np

def detect_and_remove_outliers(
    organization_models,
    X_train_vertical_FL,
    y_train,
    poisoning_budget=0.1,
    num_classes=10
):
    """
    For each participant i and each class k,
    use IsolationForest on the embedding vectors f_{i,k}.
    The top p% outliers (p = poisoning_budget) are excluded
    from the gradient updates.
    """
    excluded_indices = set()
    p = poisoning_budget  # fraction e.g. 0.1 means 10%

    # Put participant models in eval mode so we can get stable embeddings
    for model in organization_models.values():
        model.eval()

    # Number of training samples
    N = len(y_train)
    y_train_np = y_train.cpu().numpy() if y_train.is_cuda else y_train.numpy()
    
    # For each participant i, compute all embeddings once
    # embeddings_list[i] will be a tensor of shape [N, embed_dim]
    embeddings_list = {}
    for i, org_model in organization_models.items():
        # Forward pass all training samples for participant i
        with torch.no_grad():
            emb_i = org_model(X_train_vertical_FL[i])  # shape: [N, out_dim_i]
        embeddings_list[i] = emb_i.cpu().numpy()  # move to CPU if needed
    
    # For each participant i
    for i in organization_models.keys():
        emb_i = embeddings_list[i]  # shape [N, out_dim_i]

        # For each class k
        for k in range(num_classes):
            class_indices = np.where(y_train_np == k)[0]  # all indices with label k
            if len(class_indices) == 0:
                continue  # no data for that class, skip

            # Subset embeddings for class k
            emb_i_k = emb_i[class_indices, :]  # shape [num_samples_k, out_dim_i]

            # Fit Isolation Forest on emb_i_k
            if len(emb_i_k) > 1:
                iso_forest = IsolationForest(n_estimators=100, random_state=42)
                iso_forest.fit(emb_i_k)

                # Get anomaly scores (the higher, the more “outlier”)
                scores = iso_forest.score_samples(emb_i_k)  # larger => less anomalous in sklearn
                # or use iso_forest.decision_function(emb_i_k)

                # Convert to "anomaly" by negative if needed:
                # e.g. anomaly_score = -scores

                # We want the top p% with *lowest* iso_forest scores
                # Because .score_samples has higher = more normal in sklearn
                num_outliers = int(np.ceil(len(class_indices) * p))
                
                # Sort ascending so the smallest scores (most anomalous) are first
                sorted_idx = np.argsort(scores)  # ascending order
                outlier_idx_local = sorted_idx[:num_outliers]  # top p% anomalies

                # Map back to global indices in the entire dataset
                outlier_idx_global = class_indices[outlier_idx_local]
                for idx in outlier_idx_global:
                    excluded_indices.add(idx)

    print(f"[Anomaly Detection] Excluding {len(excluded_indices)} out of {N} samples")
    return excluded_indices