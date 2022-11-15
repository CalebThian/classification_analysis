from sklearn.naive_bayes import GaussianNB
from utils import getData,normalization
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import tree
from dtreeviz.trees import dtreeviz # remember to load the package
import numpy as np
from sklearn.model_selection import train_test_split

path = "./data.csv"
df,X,y = getData(path)
features = df.columns

# Split X,y using train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
X_train,X_test = normalization(df,X_train,X_test)
    
# 建立 Naive Bayes 模型
clf = GaussianNB()
# 使用訓練資料訓練模型
clf.fit(X_train, y_train)
print(np.unique(y_train,return_counts=True))
# 使用訓練資料預測分類
y_pred = clf.predict(X_test)
print(np.unique(y_pred,return_counts=True))
# 計算準確率
accuracy=clf.score(X_test, y_test)
    
print(f"Accuracy={round(accuracy*100,2)}%")
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()