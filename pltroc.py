"""
@Time    : 2022/10/21 20:58
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: pltroc.py
@Software: PyCharm
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve,auc


x = np.arange(-10, 10, 0.1)
test = pd.read_csv('./data/totalroc.csv')
sns.relplot(
    data=test, kind="line",hue="Model",style="Model",
    x="FPR", y="TPR",height=5
)


plt.savefig("roc.png",dpi=500)
plt.show()