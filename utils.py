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
    data = readFile(path)
    df = labelEncoder(data.copy())
    df = preprocess_date(df,"published date")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return df,X,y,data

## Rule(If 1 of the below rules is satisfied, recommend the course):
### 1. Subscriber > 12000
### 2. review >= 0.8*subscriber && avg.reviews >= 4.5
### 3. level = beginner and fee <= 100
### 4. level = intermediate and fee <= 250
### 5. level = expert and fee <= 500
def analysis_wrong(y_test,y_pred,X_test):
    wrong = {
        "Should not recommend": 0,
        "Not recommend subscriber > 12000": 0,
        "Not recommend review >= 0.8*subscriber && avg.reviews >= 4.5":0,
        "Not recommend level = beginner and fee <= 100":0,
        "Not recommend level = intermediate and fee <= 250":0,
        "Not recommend level = expert and fee <= 500":0
    }
    for i,(t,p) in enumerate(zip(y_test,y_pred)):
        if t != p:
            if p:
                wrong["Should not recommend"] += 1
            else:
                if X_test.iloc[i,1] > 12000:
                    wrong["Not recommend subscriber > 12000"] += 1
                elif X_test.iloc[i,4]>= 0.8*X_test.iloc[i,1] and X_test.iloc[i,5]>=4.5:
                    wrong["Not recommend review >= 0.8*subscriber && avg.reviews >= 4.5"] += 1
                elif X_test.iloc[i,6]=="Beginner" and X_test.iloc[i,3]<=100:
                    wrong["Not recommend level = beginner and fee <= 100"] += 1
                elif X_test.iloc[i,6]=="Intermediate" and X_test.iloc[i,3]<=250:
                    wrong["Not recommend level = intermediate and fee <= 250"] += 1
                elif X_test.iloc[i,6]=="Expert" and X_test.iloc[i,3]<=500:
                    wrong["Not recommend level = expert and fee <= 500"] += 1
                else:
                    print("Error classify wrong case")
            
    return wrong

def convertReview(df):
    df.iloc[:,4] = df.iloc[:,4]/df.iloc[:,1]
    return df


if __name__=="__main__":
    path = "./data.csv"
    df = readFile(path)
    df = convertReview(df)
    print(df.head(5))
    EDA(df)
    