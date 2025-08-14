import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=30, hidden_nodes=[50], num_classes=1):
        super().__init__()
        layers = []
        in_features = input_size
        for hidden_node in hidden_nodes:
            layers.append(nn.Linear(in_features, hidden_node))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # Dropout not in normal MLPs, but helped a bit with overfitting and enabled larger models
            in_features = hidden_node
        layers.append(nn.Linear(in_features, num_classes))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)