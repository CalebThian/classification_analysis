import pandas as pd
import numpy as np
from datetime import datetime
import sweetviz as sv
from sklearn import preprocessing

def readFile(path):
    data = pd.read_csv(path)
    return data

def labelEncoder(df):
    # Select categorical features
    # Method: unique value < #Row*5%  && dtypes != {float64 or int64}
    for i in range(len(df.columns)):
        if df[df.columns[i]].dtypes != "float64" and df[df.columns[i]].dtypes != "int64":
            n = len(np.unique(df[df.columns[i]]))
            if n<len(df)*0.05:
                # Label Encoding
                label_number = dict()
                for k,j in enumerate(np.unique(df[df.columns[i]])):
                    label_number[j] = k
                for r in range(len(df)):
                    df.loc[r,df.columns[i]] = label_number[df[df.columns[i]][r]]
    return df

def preprocess_date(df,column_name):
    date = df[column_name]
    timestamp = []
    for d in date:
        d_temp = datetime.strptime(d, "%Y-%m-%d")
        timestamp.append(int(datetime.timestamp(d_temp)))
    df[column_name]=timestamp
    return df

def EDA(df):
    #analyzing the dataset
    advert_report = sv.analyze(df)
    #display the report
    advert_report.show_html('EDA.html')
    
def normalization(df,X_train,X_test):
    normalizer = preprocessing.Normalizer().fit(X_train)  
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train,X_test

def getData(path):
    df = readFile(path)
    df = labelEncoder(df)
    df = preprocess_date(df,"published date")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return df,X,y

if __name__=="__main__":
    path = "./data.csv"
    df,_,_,_ = readFile(path)
    EDA(df)
    