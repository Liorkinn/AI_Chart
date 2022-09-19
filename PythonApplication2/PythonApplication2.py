import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
names = [
    "Vendor", #производитель
    "Model",  #модель
    "MYCT",   #время машинного цикла 
    "MMIN",   #мин. объем ОП
    "MMAX",   #макс объекм ОП
    "CACH",   #Кэш памяти
    "CHMIN",  #Мин кол-во каналов
    "CHMAX",  #макс кол-во каналов
    "PRP",    #опубл. относит. производительность
    "ERP",    #относительная производительность
] 

dataset = pd.read_csv(url, names=names)
array = dataset.values

X = array[:,0:9]
Y = array[:,9]
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)
j = 0

for i in names:
    print('%-15s' % (i), sep=':', end = '')
print("\n")
for j in range(30):
    for k in range(9):
        print('%-15s' % (str(X_test[j,k])), sep=':', end = '')
    print(Y_test[j], end = "\n")


print("Обработка данных массива: " , dataset.shape)

sns.set(style="whitegrid", context="notebook")
sns.pairplot(dataset[names], height=1, hue = "Vendor")
dataset[names[0:9]].hist()
plt.show()

names = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]
cm = np.corrcoef(dataset[names].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm,
    cbar=False,
    annot=True,
    square=False,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=names,
    xticklabels=names,
)
plt.show()