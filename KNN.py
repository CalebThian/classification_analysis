from sklearn.neighbors import KNeighborsClassifier
from utils import getData,normalization,analysis_wrong
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import tree
from dtreeviz.trees import dtreeviz # remember to load the package
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

path = "./data.csv"
df,X,y,ori  = getData(path)
features = df.columns

# Split X,y using train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ori_X_train, ori_X_test, ori_y_train, ori_y_test = train_test_split(ori.iloc[:,:-1], ori.iloc[:,-1], test_size=0.2, random_state=42)

# Normalization
X_train,X_test = normalization(df,X_train,X_test)

# 建立 KNN 模型
clf = KNeighborsClassifier(n_neighbors = 16)
# 使用訓練資料訓練模型
clf.fit(X_train, y_train)
print(np.unique(y_train,return_counts=True))
# 使用訓練資料預測分類
y_pred = clf.predict(X_test)
print(np.unique(y_pred,return_counts=True))
# 計算準確率
accuracy=clf.score(X_test, y_test)
    
print(classification_report(y_test, y_pred))

# Analysis Wrong
wrong = analysis_wrong(y_test,y_pred,ori_X_test)
for wrong,counts in wrong.items():
    print(f"'{wrong}': {counts}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()