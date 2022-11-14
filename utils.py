import pandas as pd

def readFile(path):
    df = pd.read_csv(path)
    print(df)
    
    
path = "./data.csv"
readFile(path)
    
    