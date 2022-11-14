from sklearn.tree import DecisionTreeClassifier
from utils import readFile

path = "./data.csv"
X,Y,features = readFile(path)
print(features)