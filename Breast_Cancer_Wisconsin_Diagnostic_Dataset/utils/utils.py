import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from torch.utils.data import DataLoader

from Breast_Cancer_Wisconsin_Diagnostic_Dataset.models.MLP import MLP
from Breast_Cancer_Wisconsin_Diagnostic_Dataset.datasets.breast_cancer_diag_dataset import BreastCancerDiagnosticDataset
import torch.nn as nn
import torch
import timeit
from tqdm import tqdm


def grid_search_MLP(X, labels, k, param_grid):
    best_accuracy = 0.0
    best_params = {}

    input_size = X.shape[1]

    for lr in param_grid['learning_rate']:
        for hidden_layer in param_grid['hidden_layers']:
            start = timeit.default_timer()
            kfold = KFold(n_splits=k, shuffle=True)
            fold_accuracies = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                train_subset = X[train_idx]
                val_subset = X[val_idx]
                train_dataset = BreastCancerDiagnosticDataset(train_subset, labels[train_idx])
                val_dataset = BreastCancerDiagnosticDataset(val_subset, labels[val_idx])

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=len(val_subset), shuffle=True)

                model = MLP(input_size=input_size, hidden_nodes=hidden_layer)
                bce_loss = nn.BCELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                model.train()
                for epoch in range(200):
                    for features, gt in train_loader:
                        output = model(features)
                        loss = bce_loss(output, gt)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    features, gt = next(iter(val_loader))
                    output = model(features)
                    pred = (output > 0.5).float()
                    acc = (pred == gt).sum().item() / len(gt)
                    fold_accuracies.append(acc)

            if np.mean(fold_accuracies) >= best_accuracy:
                best_accuracy = np.mean(fold_accuracies)
                best_params = {
                    'lr': lr,
                    'hidden_layers': hidden_layer,
                }
            stop = timeit.default_timer()
            print(f'MLP.....lr={lr}, hidden_layers=[{hidden_layer}], Acc={np.mean(fold_accuracies)} time={stop - start}')
    return best_params, best_accuracy