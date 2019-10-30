import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

class TotalGBQuantile():
    
    def __init__(self,n_estimators,learning_rate,min_samples_leaf,max_depth):
        
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        self.quantiles=[0.022750131948179195,0.15865525393145707,0.5,0.8413447460685429,0.9772498680518208]
        #keep xgboost estimator in memory 
        self.estimators = []
        
    def fit(self,X_train,y_train):
        print("training !")
        for q in tqdm(self.quantiles):
            print(f"Quantile: {q}")
            reg = GradientBoostingRegressor(loss='quantile', 
                                alpha=q,
                                n_estimators=self.n_estimators, 
                                learning_rate=self.learning_rate, 
                                min_samples_leaf=self.min_samples_leaf,
                                max_depth=self.max_depth,
                                random_state=2020)
                                
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
            dif_mean = v-v[2]
            mu = v[2]
            s = np.mean([-dif_mean[0]/2,-dif_mean[1],dif_mean[3],dif_mean[4]/2])
            mi_norm = stats.norm(mu,s)
            quant=[]
            for quantile in np.arange(1,100)/100.0 :
                quant.append(mi_norm.ppf(quantile))
            return pd.Series(quant)
 
        total_df = total_df.apply(process_row,axis=1)
        
        return total_df.values
