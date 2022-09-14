import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
interactive = st.container()

@st.cache
def get_data(filename, encoding='cp1252'):
    df = pd.read_csv(filename, encoding='cp1252')
    return df

with header:
    st.title("YY Estimation Regression Model")
    st.text('This is to demonstrate how to manage MLOps')

with dataset:
    df = get_data('VNG Training Data.csv',encoding='cp1252')
    max_record = df.shape[0]
    no_record_used = st.slider('No of record to be used',min_value=1000, max_value=max_record, value=max_record,step=1000)

    y = df.iloc[:no_record_used, 0].values
    X = df.iloc[:no_record_used, 2:15].values
    st.text("Datasets")
    st.write(df.head())
    st.text("Net YY Distribution")
    sales_dist = pd.DataFrame(df['Net_YY'].values)
    st.line_chart(sales_dist)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

with modelTraining:
  st.header('Train the model')
  sel_col,disp_col = st.columns(2)
  samples_split = sel_col.slider('What is min_samples_split', min_value=1,max_value=10,value=5,step=1)
  no_estimators = sel_col.slider('How many no_estimators',min_value=100,max_value=5000,value=500,step=100)
  # no_estimators = sel_col.selectbox('how many no_estimators',options=[100,200,300,400,500,'no limit'],index=4)
  input_feature = sel_col.text_input('which feature','test')

  params = {
        "n_estimators": no_estimators,
        "max_depth": 4,
        "min_samples_split": samples_split,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }

  reg = ensemble.GradientBoostingRegressor(**params)
  reg.fit(X_train, y_train)

  mse = mean_squared_error(y_test, reg.predict(X_test))
  st.write("The mean squared error (MSE) on test set: {:.4f}".format(mse))

with interactive:
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(df.columns[2:15])[sorted_idx])
    plt.title("Feature Importance")

    result = permutation_importance(
        reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(df.columns[2:15])[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    # plt.show()
    st.write(fig)

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    st.write(fig)
