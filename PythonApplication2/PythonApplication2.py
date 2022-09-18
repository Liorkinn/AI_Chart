import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
names = [
    "Vendor",
    "Model",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP",
    "ERP",
]
dataset = pd.read_csv(url, names=names)
array = dataset.values
print(dataset.shape)

sns.set(style="whitegrid", context="notebook")
sns.pairplot(dataset[names], height=1, hue = "Vendor")
dataset.hist()
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