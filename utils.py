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
    df = convertReview(df)
    df = preprocess_date(df,"published date")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return df,X,y,data


def analysis_wrong(y_test,y_pred,X_test):
    wrong = {
        "Should not recommend": 0,
        "Not recommend subscriber > 12000": 0,
        "Not recommend review >= 0.8*subscriber && avg.reviews >= 4.5":0,
        "Not recommend level = beginner and fee <= 100":0,
        "Not recommend level = intermediate and fee <= 250":0,
        "Not recommend level = expert and fee <= 500":0,
        "Not recommend avg.reviews >= 4.5":0,
        "Not recommend level = beginner":0,
        "Not recommend duration >= 60":0,
        "Not recommend fee <= 500":0
    }
    for i,(t,p) in enumerate(zip(y_test,y_pred)):
        if t != p:
            if p:
                wrong["Should not recommend"] += 1
            else:
                _,wrong_type = simpleRule(X_test.iloc[i,:])
                wrong["Not recommend "+wrong_type] += 1
    return wrong

def convertReview(df):
    df['reviews'] = df['reviews']/df['subscribers']
    df['reviews'] = df['reviews'].fillna(0) # Because maybe no subscriber yet
    return df

## Rule(If 1 of the below rules is satisfied, recommend the course):
### 1. Subscriber > 12000
### 2. review >= 0.8*subscriber && avg.reviews >= 4.5
### 3. level = beginner and fee <= 100
### 4. level = intermediate and fee <= 250
### 5. level = expert and fee <= 500
def ruleCheck(row_data):
    if row_data[1] > 12000:
        return 1,"subscriber > 12000"
    elif row_data[4]>= 0.8*row_data[1] and row_data[5]>=4.5:
        return 1,"review >= 0.8*subscriber && avg.reviews >= 4.5"
    elif row_data[6]=="Beginner" and row_data[3]<=100:
        return 1, "level = beginner and fee <= 100"
    elif row_data[6]=="Intermediate" and row_data[3]<=250:
        return 1, "level = intermediate and fee <= 250"
    elif row_data[6]=="Expert" and row_data[3]<=500:
        return 1, "level = expert and fee <= 500"
    else:
        return 0, "Should not recommend"
    
    
def simpleRule(row_data):
    ### 1. Subscriber > 12000
    ### 2. avg.reviews >= 4.5
    ### 3. level = beginner
    ### 4. duration > 10
    ### 5. fee <= 500
    if row_data[1] > 12000:
        return 1,"subscriber > 12000"
    elif row_data[5]>=4.5:
        return 1,"avg.reviews >= 4.5"
    elif row_data[6]=="Beginner":
        return 1, "level = beginner"
    elif row_data[8]>=60:
        return 1, "duration >= 60"
    elif row_data[3]<=500:
        return 1, "fee <= 500"
    else:
        return 0, "Should not recommend"

def jointSimpleRule(row_data):
    if row_data[1] > 8000:
        if row_data[5]>=3.5:
            if row_data[6]!="Expert":
                if row_data[8]>=60:
                    if row_data[3]<=1000:
                        return 1
    return 0
    
if __name__=="__main__":
    path = "./data.csv"
    df = readFile(path)
    df = convertReview(df)
    print(df.head(5))
    EDA(df)
    