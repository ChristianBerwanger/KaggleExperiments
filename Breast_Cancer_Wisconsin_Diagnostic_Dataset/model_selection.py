from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from Breast_Cancer_Wisconsin_Diagnostic_Dataset.datasets.breast_cancer_diag_dataset import BreastCancerDiagnosticDataset
from Breast_Cancer_Wisconsin_Diagnostic_Dataset.models.MLP import MLP
from Breast_Cancer_Wisconsin_Diagnostic_Dataset.utils import utils


def eval_models():
    # Paths for loading data and saving best model
    datapath = Path.cwd() / 'data' / 'Breast_Cancer_Diagnostic.csv'
    exp_path = Path.cwd() / 'experiments'

    df = pd.read_csv(datapath)

    # Preprocess data
    X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    y = df['diagnosis']
    le = LabelEncoder() # True,False -> 1,0
    le.fit(y)
    y=le.transform(y)
    # 80/20 Train_Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Find good parameters for various ML approaches
    # XGBoost 
    params_xg = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [2, 4, 6, 8],
        'alpha': [0, 0.5, 1, 2, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1],
    }
    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(),
        param_grid=params_xg,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    grid_search_xgb.fit(X_train, y_train)

    # Random Forest 
    params_rf = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [2, 4, 6, 8, None],
        'max_features': ['sqrt', 'log2', None],
    }
    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=params_rf,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    grid_search_rf.fit(X_train, y_train)

    # Support Vector Machine
    param_svm = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 2, 5, 10],
        'degree': [2, 3, 5],
        'gamma': [0.001, 0.01, 0.1, 0.5, 1, 'scale'],
    }
    grid_search_svm = GridSearchCV(
        estimator=svm.SVC(),
        param_grid=param_svm,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    grid_search_svm.fit(X_train_scaled, y_train)

    # Multi Layer Perceptron
    param_mlp = {
        'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05],
        'hidden_layers': [[20], [50], [50,50], [100], [100, 50], [20, 20, 20]],
    }
    mlp_best_params, mlp_best_acc = utils.grid_search_MLP(X_train_scaled, y_train,5, param_mlp)

    print('-'*25 + 'XGBoost Params' + '-'*25)
    print(grid_search_xgb.best_params_)
    print('-'*25 + 'Random Forest Params' + '-'*25)
    print(grid_search_rf.best_params_)
    print('-'*25 + 'Support Vector Machine Params' + '-'*25)
    print(grid_search_svm.best_params_)
    print('-'*25 + 'MLP Params' + '-'*25)
    print(mlp_best_params)
    
    # Final Evaluation of models

    # Already refit to the whole dataset
    xgb = grid_search_xgb.best_estimator_
    rf = grid_search_rf.best_estimator_
    svc = grid_search_svm.best_estimator_

    # Data preperation for final MLP training
    input_size = X_train.shape[1]
    hidden_layers = mlp_best_params['hidden_layers']
    # Not a lot of data -> Want to maximize train data, but keep test data same -> 65/15/20 split to introduce validation
    X_train_mlp, X_val, y_train_mlp, y_val = train_test_split(
        X_train, y_train, test_size=0.1875)
    scaler_mlp = StandardScaler()
    scaler_mlp.fit(X_train_mlp)
    X_train_mlp = scaler_mlp.transform(X_train_mlp)
    X_val_mlp = scaler_mlp.transform(X_val)
    X_test_mlp = scaler_mlp.transform(X_test)
    train_dataset = BreastCancerDiagnosticDataset(X_train_mlp, y_train_mlp)
    val_dataset = BreastCancerDiagnosticDataset(X_val_mlp, y_val)
    test_dataset = BreastCancerDiagnosticDataset(X_test_mlp, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=True)

    # Train Loop for MLP
    mlp = MLP(input_size=input_size, hidden_nodes=hidden_layers)
    bce_loss = nn.BCELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.0001, weight_decay=0.001)#mlp_best_params['lr'])
    best_val_acc = 0.0
    best_mlp_model = MLP(input_size=input_size, hidden_nodes=hidden_layers)
    # Standard Torch training loop
    for epoch in range(200):
        val_acc = 0.0
        mlp.train()
        train_acc = 0.0
        for x, y in train_loader:
            output = mlp(x)
            loss = bce_loss(output, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = (output > 0.5).float()
            train_acc += (pred == y).sum().item()
        mlp.eval()
        train_acc = train_acc / X_train_mlp.shape[0]
        with torch.no_grad():
            for x, y in val_loader:
                output = mlp(x)
                pred = (output > 0.5).float()
                val_acc += (pred == y).sum().item()
        val_acc = val_acc / len(y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_mlp_model.load_state_dict(mlp.state_dict())
            torch.save(mlp.state_dict(), exp_path / 'mlp.pkl')
        print(f"Epoch {epoch + 1}: train_acc={train_acc:.4f} - val_acc={val_acc:.4f}")
    print(f"Training Finished. Best Model Accuracy: {best_val_acc:.4f}")
    mlp.eval()

    # Test and plot results for each model
    xgb_output = xgb.predict(X_test)
    rf_output = rf.predict(X_test)
    svc_output = svc.predict(X_test_scaled)
    with torch.no_grad():
        features, gt = next(iter(test_loader))
        mlp_output = best_mlp_model(features)

    # Create Dictionary for easier looping
    model_preds = {
        'XG Boost': xgb_output,
        'Random Forest': rf_output,
        'Support Vector Machine': svc_output,
        'Multilayer Perceptron': mlp_output,
    }
    for model_name, model_pred in model_preds.items():
        if model_name == 'Multilayer Perceptron': # For handling Tensors
            model_pred = (model_pred > 0.5).float()
            y_true = gt
        else:
            y_true = y_test
        acc = (model_pred == y_true).sum().item()
        print(f"{model_name}")
        print(f" Accuracy: {acc/len(y_true):.4f}%")
        print(f" Total Number of Errors: {len(y_true)-acc}")
        plt.figure()
        #y_true = le.inverse_transform(y_true)
        #model_pred = le.inverse_transform(model_pred)
        cm = confusion_matrix(y_true, model_pred)
        custom_labels = ['Benign', 'Malicious']
        sns.heatmap(cm, annot=True, fmt='d', cmap='inferno', xticklabels=custom_labels, yticklabels=custom_labels)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

if __name__ == "__main__":
    eval_models()