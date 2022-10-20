"""
@Time    : 2022/10/9 11:36
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: explainer.py
@Software: PyCharm
"""
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from shapash import SmartExplainer
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

def encoder_minmax_values(df):
    dfencode = pd.get_dummies(df)
    columns = dfencode.columns
    print(columns)
    scaler = MinMaxScaler()
    normal = scaler.fit_transform(dfencode)
    data = DataFrame(normal,columns=columns)

    return data

df = pd.read_csv('./data/newdata4.csv')
df = df.drop('name',axis=1)
newdf = encoder_minmax_values(df)
label = newdf['label']
newdf = newdf.drop('label',axis=1)
X = newdf
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
c = XGBClassifier()

### fit model for train data
model = XGBClassifier()
model.fit(X_train,y_train)

### explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.bar(shap_values, max_display=10)
#shap.plots.scatter(shap_values[:,"Capital Gain"])
shap.plots.scatter(shap_values[:, "PRL"])
shap.plots.scatter(shap_values[:, "HDL"])
shap.plots.scatter(shap_values[:, "CRP"])
shap.plots.scatter(shap_values[:, "HGB"])
shap.plots.scatter(shap_values[:, "PLT"])
shap.plots.scatter(shap_values[:, "Pre-hemodialysis SBP"])

#shap.plots.scatter(shap_values[:,"LDL"])

#shap.plots.beeswarm(shap_values)
#shap.plots.bar(shap_values)

y_pred = model.predict(X_test)
fig, ax = plt.subplots(figsize=(15, 15))
plot_importance(model,
                height=0.5,
               ax=ax,
                max_num_features=64)
plt.savefig('totalfeatures.png',dpi = 500)
plt.show()
### model evaluate
print(classification_report(y_test, y_pred, digits=4))