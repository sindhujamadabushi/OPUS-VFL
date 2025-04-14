import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=100, target_width=11):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.target_width = target_width
        self.input_channels = 3
        self.height = 32

        self.input_dim = self.input_channels * self.height * (32 - target_width) + self.noise_dim
        self.output_dim = self.input_channels * self.height * target_width

        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, self.output_dim)

    def forward(self, x_adv, r_t):
        B = x_adv.size(0)
        x_adv_flat = x_adv.view(B, -1)
        x = torch.cat([x_adv_flat, r_t], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(B, self.input_channels, self.height, self.target_width)





# import torch



# from torch import nn
# import torch.nn.functional as F
# import sys
# sys.path.append('../../torch_utils/')

# class Generator(nn.Module):
#     def __init__(self, noise_dim=100, target_width=11):
#         super(Generator, self).__init__()
#         # For client 0, x_adv shape is [B, 3, 32, 11], so flattened dimension is 3*32*11 = 1056.
#         self.x_adv_dim = 3 * 32 * 11  
#         # The target dimension: assuming target features are also images with 3 channels, height 32, and width = target_width.
#         self.target_dim = 3 * 32 * target_width  
#         self.noise_dim = noise_dim
        
#         # Define a simple fully connected network.
#         self.fc1 = nn.Linear(self.x_adv_dim + self.noise_dim, 1024)
#         self.fc2 = nn.Linear(1024, self.target_dim)
    
#     def forward(self, x_adv, r_t):
#         """
#         x_adv: adversary's known features, shape [B, 3, 32, 11]
#         r_t: random noise vector, shape [B, noise_dim]
#         Returns:
#             Estimated target features, shape [B, 3, 32, target_width]
#         """
#         B = x_adv.size(0)
#         # Flatten the x_adv features
#         x_adv_flat = x_adv.view(B, -1)  # [B, 1056]
#         # Concatenate with the noise vector r_t
#         x = torch.cat([x_adv_flat, r_t], dim=1)  # [B, 1056 + noise_dim]
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         # Reshape the output to image dimensions: [B, 3, 32, target_width]
#         x = x.view(B, 3, 32, -1)
#         return x
    
# # class Generator(nn.Module):
# #     def __init__(self, noise_dim=100, target_width=11):
# #         super(Generator, self).__init__()
# #         self.noise_dim = noise_dim
# #         self.target_width = target_width
# #         self.image_channels = 3
# #         self.image_height = 32
# #         self.input_dim = self.image_channels * self.image_height * self.target_width + self.noise_dim

# #         self.fc1 = nn.Linear(self.input_dim, 1024)
# #         self.fc2 = nn.Linear(1024, 3 * 32 * self.target_width)

# #     def forward(self, x, noise):
# #         x = torch.flatten(x, start_dim=1)
# #         x = torch.cat((x, noise), dim=1)
# #         x = F.relu(self.fc1(x))
# #         x = torch.tanh(self.fc2(x))
# #         x = x.view(-1, 3, 32, self.target_width)
# #         return x