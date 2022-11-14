import pandas as pd
import numpy as np

def readFile(path):
    df = pd.read_csv(path)
    print(df)
    df = labelEncoder(df)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    print(df)
    return X,y,df.columns
    
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
    
path = "./data.csv"
readFile(path)
    
    