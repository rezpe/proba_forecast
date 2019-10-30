import numpy as np
import math
import pandas as pd
from tqdm import  tqdm
from sklearn.model_selection import train_test_split
from numpy.fft import fft
import datetime
import json
import numpy as np
from scipy import stats


def prepare_data_from_horizon(df, horizon=12):
    
    df=df.copy()

    # Add the previous values 
    sel = np.concatenate([[1,2,3,4],
                         [12],
                         24*np.arange(1,9),
                         12+24*np.arange(1,9)])  
    sel=np.concatenate([sel,sel-1,sel+1])  
    
    for i in sel:
        if (i>=horizon):
            df["NO2 - "+str(i)] = df["NO2"].shift(i)
            #df["NO2d - "+str(i)] = df["NO2"].shift(i)-df["NO2"].shift(i+24)

    calcols = ['Calendar.Festivo.EqiNoc',
 'Calendar.Festivo.InmCns',
 'Calendar.Festivo.NavAnu',
 'Calendar.Festivo.PriVer',
 'Calendar.Festivo.SemSan',
 'Calendar.Festivo.VrgAgo',
 'Calendar.OprRetorno',
 'Calendar.OprSalida.Vispera',
 'Calendar.School.vacaciones.semana_santa',
 'Calendar.School.vacaciones.verano',
 'Calendar.Festivo.FesNov',
 'Calendar.NocheNav.NocheBuena',
 'Calendar.NocheNav.NocheVieja',
 'Calendar.OprSalida.PriNoLab',
 'Calendar.PuenteLab',
 'Calendar.School.intensiva.fin_curso',
 'Calendar.School.intensiva.inicio_curso',
 'Calendar.School.intensiva.navidad',
 'Calendar.School.no_lectivo.otros',
 'Calendar.School.vacaciones.navidad']

    for col in calcols:
        for i in 24*np.array([1,2,7]):
            if (i>=horizon):
                df[col+" - "+str(i)] = df[col].shift(i)
        
    # We take the old O3
    df["O3"]=np.log1p(df["O3"])
    for i in 24*np.arange(1,4):
        if (i>=horizon):
            df["O3 - "+str(i)] = df["O3"].shift(i)
            
    df=df.dropna()
    
    X=df[list(set(df.columns)-set(['date','day',"Index","NO2","O3"]))]
    y=df["NO2"]
        
    # Get the training and test data
    training_index = df[df["date"].dt.year<2017].index
    test_index = df[(df["date"].dt.year>2016)&((df["date"]-datetime.timedelta(hours=horizon)).dt.hour==10)].index
    
    # We must remove from test
    dup = df["date"][df["date"].duplicated()].index
    test_index = test_index[~test_index.isin(dup)]

    X_train =  X.loc[training_index]
    X_test = X.loc[test_index]
    y_train = y.loc[training_index]
    y_test =y.loc[test_index]
    
    return X_train, X_test, y_train, y_test

#"data/dataEscAgui.csv"
def get_data(path):

    df=pd.read_csv("data/dataEscAgui.csv",sep=" ").reset_index()
    df=df.rename(columns={"level_0":"date","level_1":"day"})
    df["date"]=pd.to_datetime(df["date"],format="%Y-%m-%d %H:%M:%S")
    df=df.sort_values("date")
    
    df=df.interpolate()

    # Fourier (The frequencies were obtained in a different notebook)

    freqs = [2922,1461,209,1465,4]
    l = 35064
    n = np.arange(len(df))
    fcols = []
    for f in freqs:
        df["c"+str(f)]=np.cos(n*2*np.pi*f/l)
        fcols.append("c"+str(f))
        df["s"+str(f)]=np.cos(n*2*np.pi*f/l)
        fcols.append("s"+str(f))


    #log NO2
    df["NO2"]=np.log1p(df["NO2"])

    calcols = ['Calendar.Festivo.EqiNoc',
 'Calendar.Festivo.InmCns',
 'Calendar.Festivo.NavAnu',
 'Calendar.Festivo.PriVer',
 'Calendar.Festivo.SemSan',
 'Calendar.Festivo.VrgAgo',
 'Calendar.OprRetorno',
 'Calendar.OprSalida.Vispera',
 'Calendar.School.vacaciones.semana_santa',
 'Calendar.School.vacaciones.verano',
 'Calendar.Festivo.FesNov',
 'Calendar.NocheNav.NocheBuena',
 'Calendar.NocheNav.NocheVieja',
 'Calendar.OprSalida.PriNoLab',
 'Calendar.PuenteLab',
 'Calendar.School.intensiva.fin_curso',
 'Calendar.School.intensiva.inicio_curso',
 'Calendar.School.intensiva.navidad',
 'Calendar.School.no_lectivo.otros',
 'Calendar.School.vacaciones.navidad']

    

    # log ECMWF
    mcols = ['MACC.NO2.Lon35635_Lat4045_Lead0',
            'MACC.NO2.Lon35635_Lat4045_Lead1', 'MACC.NO2.Lon35635_Lat4045_Lead2',
            'MACC.NO2.Lon35625_Lat4045_Lead0', 'MACC.NO2.Lon35625_Lat4045_Lead1',
            'MACC.NO2.Lon35625_Lat4045_Lead2', 'MACC.NO2.Lon35635_Lat4035_Lead0',
            'MACC.NO2.Lon35635_Lat4035_Lead1', 'MACC.NO2.Lon35635_Lat4035_Lead2',
            'MACC.NO2.Lon35625_Lat4035_Lead0', 'MACC.NO2.Lon35625_Lat4035_Lead1',
            'MACC.NO2.Lon35625_Lat4035_Lead2']

    df[mcols]=np.log1p(df[mcols])
    
    return df 