import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

from all_data_preprocessing_update import *

from app_quantile_regression.metrics import *
from app_quantile_regression.mlp import *
from app_quantile_regression.metrics import *
from app_quantile_regression.knn import *
from app_quantile_regression.qrf import *
from app_quantile_regression.linear import *
from app_quantile_regression.lgba import *

from ngboost import NGBRegressor
from ngboost.distns import Normal

from recorder import *

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

from datetime import datetime,timedelta

folder="mlp_new_training"

if not os.path.exists(folder):
    os.makedirs(folder)

df = get_data("./2018_2019_data/28079008.csv")

for horizon in tqdm(range(1,60)):

    X, y = prepare_data_from_horizon(df,horizon)

    prev = open("status.txt","r").read()
    logout = open("status.txt","w")
    time = datetime.now()
    logout.write(prev+f"\n{horizon}:{time}")
    logout.close()

    kf = KFold(5,shuffle=True)
    for train_index, test_index in kf.split(X):

        train_index = X.index.values[train_index]
        test_index = X.index.values[test_index] 
        
        # Filter the test index when prediction time is 10:00
        ten_index = df[(df["DATE"]-timedelta(hours=horizon)).dt.hour==10].index
        test_index_10 = test_index[pd.Series(test_index).isin(ten_index)]
        
        # We retrieve the indexes that are related to the test indexes according to our AR model
        sel = np.concatenate([[1,2,3,4],
                            [12],
                            24*np.arange(1,9),
                            12+24*np.arange(1,9)])  
        sel=np.concatenate([sel,sel-1,sel+1]) 
        
        all_index_related_test = set([])
        for i in sel:
            all_index_related_test |= set(test_index_10+i)
        
        train_index_CV = train_index[pd.Series(train_index).isin(list(all_index_related_test))]
        
        X_train = X.loc[train_index_CV]
        y_train = y.loc[train_index_CV]
        
        X_test = X.loc[test_index_10]
        y_test = y.loc[test_index_10]
        
        scaler = RobustScaler()
        # Fit the scaler on the training features and transform these in one go
        X_train_std = scaler.fit_transform(X_train)
        # Scale the test set
        X_test_std = scaler.transform(X_test)
        
        lin = LinearRegression()
        lin.fit(X_train_std,y_train)

        dif_train = y_train-lin.predict(X_train_std)
        dif_test = y_test-lin.predict(X_test_std)

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
        save_result([horizon,"QKNNL",results,1],f"unit_{horizon}",folder)

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
        ngb = NGBRegressor(Dist=Normal, verbose=True,n_estimators=800)
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

        start = datetime.now().timestamp()
        qreg = QuantileKNN(n_neighbors=50)
        qreg.fit(X_train_std,y_train.values)
        preds = qreg.predict(X_test_std)
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
        qreg = TotalLGBQuantile(n_estimators=1900,max_depth=16)
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
        qreg = TotalLGBQuantile(n_estimators=1900,max_depth=16)
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
                        
