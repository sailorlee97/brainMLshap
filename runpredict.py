"""
@Time    : 2022/10/6 10:03
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: runpredict.py
@Software: PyCharm
"""
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

df = pd.read_csv('./data/newdata4.csv')
feature_name = ['age', 'Duration of dialysis', 'WBC', 'PLT', 'HGB', 'NE', 'LY', 'HCT',
       'CRP', 'NRL', 'PRL', 'ALT', 'AST', 'TP', 'ALB', 'BUN', 'Scr', 'cysc',
       'UA', 'TG', 'TC', 'LDL', 'HDL', 'K', 'NA', 'Ca', 'P',
       'Calcium-phosphorus product', 'eGFR', 'Total anticoagulant',
       'Blood flow rate', 'Pre-hemodialysis SBP', 'Pre-hemodialysis DBP',
       'Post-hemodialysis SBP', 'Post-hemodialysis DBP', 'label', 'sex_Female',
       'sex_male', 'Hypertension _no', 'Hypertension _yes', 'DM_no', 'DM_yes',
       'Polycystic kidney_no', 'Polycystic kidney_yes',
       'Hemodialysis Vascular Access_arteriovenous fistula',
       'Hemodialysis Vascular Access_artificial blood vessel',
       'Hemodialysis Vascular Access_central venous catheter']
#categorical_features = [col for col in df.columns if df[col].dtype == 'object']
#print(categorical_features)

#encoder = OrdinalEncoder(
#    cols=categorical_features,
#    handle_unknown='ignore',
#    return_df=True).fit(df)

#X_df=encoder.transform(df)
#print(X_df)

df = df.drop('name',axis=1)
listencode = ['sex','Hypernsion grade III','Polycystic kidney','DM','Hemodialysis Vascular Access']

#dfnormal = dfnormal.join(dfnormalencode)
#dfnormal =dfnormal.drop(listencode,axis = 1,inplace = True)
#for i in listencode:
 #   if i in columns:
  #      dfnormal = dfnormal.join(pd.get_dummies(dfnormal.i))
 #       dfnormal = dfnormal.drop(i,axis=1)

def encoder_minmax_values(df):
    dfencode = pd.get_dummies(df)
    columns = dfencode.columns
    print(columns)
    scaler = MinMaxScaler()
    normal = scaler.fit_transform(dfencode)
    data = DataFrame(normal,columns=columns)

    return data

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

#explainer = shap.Explainer(model, X)
#shap_values = explainer(X)
#shap.plots.bar(shap_values)

#shap.plots.beeswarm(shap_values)

explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]
print(f"Explainer expected value: {expected_value}")

select = range(20)
features = X_test.iloc[select]
features_display = X_test.loc[features.index]

for i  in range(20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(features)[i]
    shap.force_plot(expected_value, shap_values, features_display.iloc[i].apply(lambda x: round(x, 4)), matplotlib=True)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     shap_values = explainer.shap_values(features)[1]
#     shap_interaction_values = explainer.shap_interaction_values(features)
# if isinstance(shap_interaction_values, list):
#     shap_interaction_values = shap_interaction_values[1]
#     # Our naive cutoff point is zero log odds (probability 0.5).

# y_pred = model.predict(X_test)
# shap.force_plot(expected_value, shap_values, features_display.iloc[1], matplotlib=True)

# misclassified = y_pred != y_test
# shap.decision_plot(expected_value, shap_values, features_display, link='logit', highlight=misclassified)

# shap.force_plot(expected_value, shap_values, X_test[0],feature_name,text_rotation=30,matplotlib=True)
# shap.decision_plot(expected_value, shap_values, features_display)
# ### plot feature importance
# #fig, ax = plt.subplots(figsize=(15, 15))
# #plot_importance(model,
# #                height=0.5,
# #                ax=ax,
# #                max_num_features=64)
# plt.show()

### make prediction for test data


### model evaluate
print(classification_report(y_test, y_pred, digits=4))
