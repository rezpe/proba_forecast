import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

from all_data_preprocessing import *

from app_quantile_regression.metrics import *
from app_quantile_regression.knn import *
from app_quantile_regression.qrf import *
from app_quantile_regression.linear import *
from app_quantile_regression.lgba import *

from recorder import *

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

df = get_data("https://gist.githubusercontent.com/rezpe/ee4d91dbe1f6104f2120aa94b8e5f60c/raw/18fc2daa4f04ded39d8b81e4a4316e424dc82d43/dataEscAgui.csv")
folder="destination_folder"

if not os.path.exists(folder):
    os.makedirs(folder)

for horizon in tqdm(range(1,60)):

    X_train, X_test, y_train, y_test = prepare_data_from_horizon(df,horizon)
    
    scaler = RobustScaler()
    # Fit the scaler on the training features and transform these in one go
    X_train_std = scaler.fit_transform(X_train)
    # Scale the test set
    X_test_std = scaler.transform(X_test)
    
    lin = LinearRegression()
    lin.fit(X_train_std,y_train)

    dif_train = y_train-lin.predict(X_train_std)
    dif_test = y_test-lin.predict(X_test_std)
    
    print("Models")

    start = datetime.now().timestamp()
    qreg = QuantileKNN(n_neighbors=50)
    qreg.fit(X_train_std,dif_train.values)
    pred_difs = qreg.predict(X_test_std)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std) 
    end = datetime.now().timestamp()
    results = evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QKNNL",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    qreg = QuantileKNN(n_neighbors=50)
    qreg.fit(X_train.values,y_train.values)
    preds = qreg.predict(X_test.values)
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QKNN",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    qreg = RandomForestQuantileRegressor(n_estimators=50,min_samples_leaf=50,max_depth=6)
    qreg.fit(X_train,dif_train)
    pred_difs = qreg.predict(X_test)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std) 
    end = datetime.now().timestamp()
    results=evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values)
    results["duration"]=end-start 
    save_result([horizon,
                    "QRFL",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp() 
    qreg = RandomForestQuantileRegressor(n_estimators=500,min_samples_leaf=50,max_depth=6)
    qreg.fit(X_train,y_train)
    preds = qreg.predict(X_test)
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QRF",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    qreg = TotalLinearQuantile()
    qreg.fit(X_train_std,y_train)
    preds = qreg.predict(X_test_std)
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QLR",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    qreg = TotalLGBQuantile(n_estimators=1000,max_depth=12)
    qreg.fit(X_train_std,y_train)
    preds = qreg.predict(X_test_std)
    end = datetime.now().timestamp()
    results=evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QGB",
                    results,
                    1],f"unit_{horizon}",folder)

    start = datetime.now().timestamp()
    qreg = TotalLGBQuantile(n_estimators=1000,max_depth=12)
    qreg.fit(X_train,dif_train)
    pred_difs = qreg.predict(X_test)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std) 
    end = datetime.now().timestamp()
    results=evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values)
    results["duration"]=end-start
    save_result([horizon,
                    "QGBL",
                    results,
                    1],f"unit_{horizon}",folder)

                    
