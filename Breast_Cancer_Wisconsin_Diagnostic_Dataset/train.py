import pandas as pd
import argparse
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
from pathlib import Path
import utils.utils as utils
from xgboost import XGBClassifier

def train_model():
    model_name = 'xgboost'

    datapath = Path.cwd() / 'data' / 'Breast_Cancer_Diagnostic.csv'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path('experiments' + '/logs/' + model_name +'_' + current_time)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    df = pd.read_csv(datapath)

    # Preprocess data
    X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    y = df['diagnosis']
    le = LabelEncoder()
    le.fit(y)
    y=le.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # XGBoost
    xg_clf = XGBClassifier()
    xg_clf.fit(X_train, y_train)
if __name__ == "__main__":
    train_model()