"""
File: models.py
------------------
This file holds the torch.nn modules. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb


num_classes = 2  # binary: disease vs. non-diseased 
num_experts = 6  # one for each domain (skin color) 


# --- Define the Expert Network ---
class Expert(nn.Module):
    def __init__(self, num_classes=2):
        super(Expert, self).__init__()
        # Simple CNN for image classification
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, num_classes)
        )

    def forward(self, x):
        return self.network(x)



# --- Define the Gating Network ---
class GatingNetwork(nn.Module):
    def __init__(self, num_experts=6):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)




# --- Define Mixture of Experts ---
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, num_classes=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(num_classes) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(num_experts)

    def forward(self, x, aux=True):
        # aux means auxiliary 
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gating_distributions = self.gating_network(x)
        pdb.set_trace()
        output = torch.sum(gating_distributions.unsqueeze(2) * expert_outputs, dim=1)

        if aux:
            output_dict = {'output': output, 'expert_outputs': expert_outputs, 'gating_distributions': gating_distributions}
            return output_dict
        else:
            return output



