import numpy as np

def o2s_mechanism_with_shapley(shapley_values, bid_prices, organization_num, features_per_org, budget):
   
    org_values = []
    org_costs = []
    org_feature_map = []

    start = 0
    for i in range(organization_num):
        end = start + features_per_org[i]
        org_feature_map.append(list(range(start, end)))
        org_values.append(np.sum(shapley_values[start:end]))
        org_costs.append(np.sum(bid_prices[start:end]))
        start = end

    # 0/1 Knapsack: maximize value within cost ≤ budget
    best_value = 0
    best_selection = []

    from itertools import combinations
    for r in range(1, organization_num + 1):
        for org_combo in combinations(range(organization_num), r):
            total_cost = sum(org_costs[i] for i in org_combo)
            total_value = sum(org_values[i] for i in org_combo)
            if total_cost <= budget and total_value > best_value:
                best_value = total_value
                best_selection = org_combo

    selected_organizations = list(best_selection)
    selected_features = [f for org in selected_organizations for f in org_feature_map[org]]
    rewards = np.zeros_like(shapley_values)
    for f in selected_features:
        rewards[f] = bid_prices[f]  # Reward is bid price for each selected feature

    return selected_organizations, selected_features, rewards