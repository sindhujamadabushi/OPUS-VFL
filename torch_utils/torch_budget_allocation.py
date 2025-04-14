import numpy as np
import math

   
    
def calculate_rewards(contributions, fixed_resource_allocation_with_cl, fixed_budget_allocation_from_server, gamma):
    # Step 1: contributions / fixed_resource_allocation_with_cl
    ratios = contributions / np.array(fixed_resource_allocation_with_cl)

    # Step 2: interim_rewards = contributions / (contributions + fixed_resource_allocation_with_cl)
    interim_rewards = contributions / (contributions + gamma*np.array(fixed_resource_allocation_with_cl))

    # Step 3: rewards = interim_rewards / total_interim_rewards * fixed_budget_allocation_from_server
    total_interim_rewards = np.sum(interim_rewards)
    rewards = interim_rewards / total_interim_rewards * fixed_budget_allocation_from_server
    utility = rewards - fixed_resource_allocation_with_cl
    return rewards, utility

# def calculate_rewards(contributions, fixed_resource_allocation_with_cl, fixed_budget_allocation_from_server, gamma):
#     # Step 1: contributions / fixed_resource_allocation_with_cl
#     ratios = contributions / np.array(fixed_resource_allocation_with_cl)

#     # Step 2: interim_rewards = contributions / (contributions + fixed_resource_allocation_with_cl)
#     interim_rewards = contributions / (contributions + gamma)

#     # Step 3: rewards = interim_rewards / total_interim_rewards * fixed_budget_allocation_from_server
#     total_interim_rewards = np.sum(interim_rewards)
#     rewards = interim_rewards / total_interim_rewards * fixed_budget_allocation_from_server
#     utility = rewards - fixed_resource_allocation_with_cl
#     return rewards, utility

