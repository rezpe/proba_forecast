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

    X=data[list(set(data.columns)-set(['DATE',"NO2","O3","day","dt_date"]))]
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
    ## # Day
    data["day"]=data["DATE"].dt.strftime("%Y-%m-%d")

    from datetime import date, timedelta, datetime

    def get_alldates(starts,ends):
        sdate = datetime.strptime(starts,"%Y-%m-%d")
        edate = datetime.strptime(ends,"%Y-%m-%d") 

        delta = edate - sdate       # as timedelta

        all_dates = []
        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            all_dates.append(day.strftime("%Y-%m-%d"))

        return all_dates

    # Escolar

    cal_escolar = pd.read_csv("./cal_extract/cal_f_escolar.csv")

    ## No cogemos los dias con COVID 19
    cal_escolar = cal_escolar[~cal_escolar["co_period"].str.contains("covid19")]

    all_cal_days = []
    for i,row in cal_escolar.iterrows():
        all_dates = get_alldates(row["dt_start"],row["dt_stop"])

        def process_date(date,typed):
            return {"day":date,typed:"1"}

        for element in map(lambda date: process_date(date,"Calendar.School."+row["co_type"]+"."+row["co_period"]),all_dates):
            all_cal_days.append(element)

    all_cal_days_df = pd.DataFrame(all_cal_days).fillna(0)
    
    for col in all_cal_days_df.columns:
        if col!="day":
            all_cal_days_df[col]=all_cal_days_df[col].astype(int)

    ## Otros Festivos
    """
    Calendar.Festivo.EqiNoc
    Calendar.Festivo.FesNov
    Calendar.Festivo.InmCns
    Calendar.Festivo.NavAnu
    Calendar.Festivo.PriVer
    Calendar.Festivo.SemSan
    Calendar.Festivo.VrgAgo
    Calendar.NocheNav.NocheBuena
    Calendar.NocheNav.NocheVieja
    Calendar.OprRetorno
    Calendar.OprSalida.PriNoLab
    Calendar.OprSalida.Vispera
    """ 
    cal_festivos = pd.read_csv("./cal_extract/cal_f_festivos.csv")
    
    
    ## For each type we take the OR of the columns
    holiday_list = """Festivo.EqiNoc
    Calendar.Festivo.FesNov
    Calendar.Festivo.InmCns
    Calendar.Festivo.NavAnu
    Calendar.Festivo.PriVer
    Calendar.Festivo.SemSan
    Calendar.Festivo.VrgAgo
    Calendar.NocheNav.NocheBuena
    Calendar.NocheNav.NocheVieja
    Calendar.OprRetorno
    Calendar.OprSalida.PriNoLab
    Calendar.OprSalida.Vispera""".replace("Calendar.","").lower().replace(".","_").split("\n")

    def get_festivos(name,row):
        cols = row.index[row.index.str.contains(name)]
        res = row[cols].sum()
        return res

    res_festivos_df = pd.DataFrame()
    for name in holiday_list:
        print(name)
        res_festivos_df["Calendar."+name]=cal_festivos.apply(lambda row:get_festivos(name,row),axis=1).astype(int).fillna(0)

    res_festivos_df["day"]=cal_festivos["dt_date"]
    
    data = data.merge(all_cal_days_df,on="day",how="left")
    data = data.merge(res_festivos_df,on="day",how="left")
    
    data=data.fillna(0)
    
    return data