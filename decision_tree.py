from sklearn.tree import DecisionTreeClassifier
from utils import getData,analysis_wrong,featureImportance
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
df,X,y,ori = getData(path)
features = df.columns

# Split X,y using train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ori_X_train, ori_X_test, ori_y_train, ori_y_test = train_test_split(ori.iloc[:,:-1], ori.iloc[:,-1], test_size=0.2, random_state=42)
    
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
    
print(classification_report(y_test, y_pred))

# Analysis Wrong
wrong = analysis_wrong(y_test,y_pred,ori_X_test)
for wrong,counts in wrong.items():
    print(f"'{wrong}': {counts}")

# Print decision tree
text_representation = tree.export_text(clf,feature_names = list(features)[:-1])
with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)
    
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

featureImportance(clf,features)

# Visualize the best tree
viz = dtreeviz(clf, X, y,
                target_name="target",
                feature_names=features,
                class_names=['0','1'])

viz.save("decision_tree.svg")