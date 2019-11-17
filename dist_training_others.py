import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

from all_data_preprocessing import *

from app_quantile_regression.metrics import *
from app_quantile_regression.mlp import *

from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from ngboost.distns import Normal

from recorder import *

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

folder="destination_folder"

if not os.path.exists(folder):
    os.makedirs(folder)

for horizon in tqdm(range(1,60)):

    df = get_data("https://gist.githubusercontent.com/rezpe/ee4d91dbe1f6104f2120aa94b8e5f60c/raw/18fc2daa4f04ded39d8b81e4a4316e424dc82d43/dataEscAgui.csv")
    X_train, X_test, y_train, y_test = prepare_data_from_horizon(df,horizon)
    
    scaler = RobustScaler()
    # Fit the scaler on the training features and transform these in one go
    X_train_std = scaler.fit_transform(X_train)
    # Scale the test set
    X_test_std = scaler.transform(X_test)
    
    print("Models")

    start = datetime.now().timestamp()
    qreg = MLPQuantile()
    qreg.fit(X_train_std,y_train)
    preds = qreg.predict(X_test_std)
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "MLP",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True,
              verbose=True,n_estimators=1500)
    ngb.fit(X_train_std, y_train.values)
    Y_dists = ngb.pred_dist(X_test_std)
    a=pd.DataFrame()
    for i in np.arange(1,100):
        a[i]=Y_dists.ppf(i/100)
    preds = a.values
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "NGBOOST",
                    results,
                    1],f"unit_{horizon}",folder)

                    
