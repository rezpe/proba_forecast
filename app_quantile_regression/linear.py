import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from scipy import stats

from mlinsights.mlmodel import QuantileLinearRegression

class TotalLinearQuantile():
    
    def __init__(self):
        
        #keep xgboost estimator in memory 
        self.estimators = []
        self.quantiles=[0.022750131948179195,0.15865525393145707,0.5,0.8413447460685429,0.9772498680518208]
        
    def fit(self,X_train,y_train):
        print("training")
        for q in tqdm(self.quantiles):
            reg = QuantileLinearRegression(quantile=q)
            reg.fit(X_train, y_train)
            self.estimators.append(reg)
        print("Done")
        
        
    def predict(self,X):
        predictions_gbr = []
        print("predicting")
        for reg in tqdm(self.estimators):
            predictions_gbr.append(reg.predict(X))
         
        total_pred={}
        for i in range(len(predictions_gbr)):
            total_pred[i]=predictions_gbr[i]
            
        total_df=pd.DataFrame(total_pred)

        def process_row(row):
            v = row.values
            dif_mean = np.abs(v-v[2])
            mu = v[2]
            s = np.mean([dif_mean[0]/2,dif_mean[1],dif_mean[3],dif_mean[4]/2])
            mi_norm = stats.norm(mu,s)
            quant=[]
            for quantile in np.arange(1,100)/100.0 :
                quant.append(mi_norm.ppf(quantile))
            
            return pd.Series(quant)
            
        total_df = total_df.apply(process_row,axis=1)
           
        return total_df.values

