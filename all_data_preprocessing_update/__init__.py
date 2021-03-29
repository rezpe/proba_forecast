import numpy as np
import math
import pandas as pd
from tqdm import  tqdm
from sklearn.model_selection import KFold
from numpy.fft import fft
import datetime
import json
import numpy as np
from scipy import stats

def prepare_data_from_horizon(df, horizon=12):
    
    data=df.copy()

    sel = np.concatenate([[1,2,3,4],
                         [12],
                         24*np.arange(1,9),
                         12+24*np.arange(1,9)])  
    sel=np.concatenate([sel,sel-1,sel+1])  
    
    ## lagged NO2 values
    for i in sel:
        if (i>=horizon):
            data["NO2 - "+str(i)] = data["NO2"].shift(i)

    ## lagged O3 values
    for i in 24*np.arange(1,4):
        if (i>=horizon):
            data["O3 - "+str(i)] = data["O3"].shift(i)

    ## Remove empty values
    data=data.dropna()

    X=data[list(set(data.columns)-set(['DATE',"NO2","O3"]))]
    y=data["NO2"]
  
    return X, y

#"data/dataEscAgui.csv"
def get_data(path):

    df = pd.read_csv(path,sep=";")

    # Prepare data
    data = df[["DATE","SPA.NO2","SPA.O3","MACC.NO2"]].copy()
    data["DATE"]=pd.to_datetime(data["DATE"],format="%Y-%m-%d %H:%M:%S")
    data = data.sort_values("DATE")
    data.columns = ["DATE","NO2","O3","CAMS"]

    ## Remove everything from 2020
    data=data[data["DATE"].astype(str)<"2020"]

    ## Fourier Columns
    freqs = [2922,1461,209,1465,4]
    l = 35064
    n = np.arange(len(data))
    fcols = []
    for f in freqs:
        data["c"+str(f)]=np.cos(n*2*np.pi*f/l)
        fcols.append("c"+str(f))
        data["s"+str(f)]=np.cos(n*2*np.pi*f/l)
        fcols.append("s"+str(f))

    data["NO2"]=np.log1p(data["NO2"])
    data["O3"]=np.log1p(data["O3"])
    data["CAMS"]=np.log1p(data["CAMS"])

    ## Calendar Variables 
    ## Calendar Variables do not bring better results and therefore
    ## removed
    
    return data