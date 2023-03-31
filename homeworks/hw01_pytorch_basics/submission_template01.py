import numpy as np
import torch
import torch.nn as nn


def create_model():
    return nn.Sequential(nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 16),
                        nn.ReLU(),
                        nn.Linear(16, 10))

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    return total_params
    