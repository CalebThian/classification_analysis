from sklearn.tree import DecisionTreeClassifier
from utils import readFile
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


path = "./data.csv"
X,y,features = readFile(path)

# Split X,y using KFold
kf = KFold(n_splits=5)


# Train-test-evaluate in for loop
for train_index, test_index in kf.split(X):
    # Split Train and Test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 建立 DecisionTreeClassifier 模型
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=10, random_state=10)
    # 使用訓練資料訓練模型
    tree.fit(X_train, y_train)
    # 使用訓練資料預測分類
    y_pred = tree.predict(X_test)
    # 計算準確率
    accuracy = tree.score(X_test, y_test)
    print(f"accuracy={round(accuracy*100,2)}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()