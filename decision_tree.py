from sklearn.tree import DecisionTreeClassifier
from utils import getData
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
    
# 建立 Decision Tree 模型
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=10, random_state=10)
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

# Visualize the best tree
viz = dtreeviz(clf, X, y,
                target_name="target",
                feature_names=features,
                class_names=['0','1'])

viz.save("decision_tree.svg")