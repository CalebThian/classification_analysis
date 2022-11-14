import pandas as pd

def readFile(path):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X,y,df.columns
    
    
path = "./data.csv"
readFile(path)
    
    