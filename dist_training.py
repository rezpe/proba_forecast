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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

df = get_data("data/dataEscAgui.csv")
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
    
    result={}
    result["horizon"]=horizon
    
    print("Models")

    qreg = QuantileKNN(n_neighbors=50)
    qreg.fit(X_train_std,dif_train.values)
    pred_difs = qreg.predict(X_test_std)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std) 
    save_result([horizon,
                    "QKNNL",
                    evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

    qreg = QuantileKNN(n_neighbors=50)
    qreg.fit(X_train.values,y_train.values)
    preds = qreg.predict(X_test.values)
    save_result([horizon,
                    "QKNN",
                    evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

    qreg = RandomForestQuantileRegressor(n_estimators=50,min_samples_leaf=50,max_depth=6)
    qreg.fit(X_train,dif_train)
    pred_difs = qreg.predict(X_test)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std)  
    save_result([horizon,
                    "QRFL",
                    evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)
    
    qreg = RandomForestQuantileRegressor(n_estimators=500,min_samples_leaf=50,max_depth=6)
    qreg.fit(X_train,y_train)
    preds = qreg.predict(X_test)
    save_result([horizon,
                    "QRF",
                    evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

    qreg = TotalLinearQuantile()
    qreg.fit(X_train_std,y_train)
    preds = qreg.predict(X_test_std)
    save_result([horizon,
                    "QLR",
                    evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

    qreg = TotalLGBQuantile(n_estimators=1000,max_depth=12)
    qreg.fit(X_train_std,y_train)
    preds = qreg.predict(X_test_std)
    save_result([horizon,
                    "QGB",
                    evaluate((np.exp(preds)-1),(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

    qreg = TotalLGBQuantile(n_estimators=1000,max_depth=12)
    qreg.fit(X_train,dif_train)
    pred_difs = qreg.predict(X_test)
    a=pd.DataFrame(pred_difs)
    for col in a.columns:
        a[col] += lin.predict(X_test_std) 
    save_result([horizon,
                    "QGBL",
                    evaluate((np.exp(a)-1).values,(np.exp(y_test)-1).values),
                    1],f"unit_{horizon}",folder)

                    
