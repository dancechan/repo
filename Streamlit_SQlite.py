from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import sqlite3

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
interactive = st.container()
prediction = st.container()

@st.cache
def get_data(filename, encoding='cp1252'):
    df = pd.read_csv(filename, encoding='cp1252')
    return df

df = get_data('VNG Training Data.csv',encoding='cp1252')
max_record = df.shape[0]
y = df.iloc[:, 0].values
X = df.iloc[:, 2:14].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        }
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))

with header:
    st.title("YY Estimation Regression Model")
    # st.text('This is to demonstrate how to manage MLOps')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dataset","Model Training","Evaluation", "Prediction"],
        default_index = 3,
    )
if selected == "Dataset":

    with dataset:

        no_record_used = st.slider('Total no. of records',min_value=1000, max_value=max_record, value=max_record,step=1000)

        y = df.iloc[:no_record_used, 0].values
        X = df.iloc[:no_record_used, 2:14].values
        st.text("Datasets")
        st.write(no_record_used)
        st.text("Net YY Distribution")
        sales_dist = pd.DataFrame(df.iloc[:no_record_used,0].values,columns=["Net YY"])
        st.line_chart(sales_dist)
        AgGrid(df.iloc[:10,:])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

if selected == "Model Training":

    with modelTraining:
      st.header('Train the model')
      sel_col,disp_col = st.columns(2)
      samples_split = sel_col.slider('What is min_samples_split', min_value=1,max_value=10,value=5,step=1)
      no_estimators = sel_col.slider('How many no_estimators',min_value=100,max_value=5000,value=500,step=100)
      test_percent = sel_col.selectbox('What portion of record for test',options=[0.1,0.2,0.3,0.4],index=1)
      lr = sel_col.slider('What is learning rate', min_value =0.01, max_value=0.9, value=0.01)

      params = {
            "n_estimators": no_estimators,
            "max_depth": 4,
            "min_samples_split": samples_split,
            "learning_rate": lr,
            "loss": "squared_error",
        }
      max_record = df.shape[0]
      no_record_used = st.slider('No of record to be used for training and testing', min_value=1000, max_value=max_record, value=max_record,
                                 step=1000)

      y = df.iloc[:no_record_used, 0].values
      X = df.iloc[:no_record_used, 2:14].values

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=13)
      reg = ensemble.GradientBoostingRegressor(**params)
      reg.fit(X_train, y_train)

      mse = mean_squared_error(y_test, reg.predict(X_test))
      rmse = np.sqrt(mse)
      mae = mean_absolute_error(y_test, reg.predict(X_test))
      mape = mean_absolute_percentage_error(y_test, reg.predict(X_test))
      st.write("The mean squared error (MSE) on test set: {:.4f}".format(mse))
      st.write("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
      st.write("The mean absolute error (MAE) on test set: {:.4f}".format(mae))
      st.write("The mean absolute percentage error (MAPE) on test set: {:.4f}%".format(mape*100))

if selected == "Evaluation":
    with interactive:
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
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

if selected == "Prediction":
    with st.form("my_form"):
        customer = st.selectbox('What customer?', options=('BB', 'CT', 'DIL', 'PF'), index=1)
        pattern = st.radio(
            "What is pattern?",
            ('Solid', 'Stripe', 'Check'),index=0,horizontal=True)
        repeat_x = st.number_input('Repeat in Weft', min_value=0.0, max_value=5.0,value=0.0)
        repeat_y = st.number_input('Repeat in Warp', min_value=0.0, max_value=5.0,value=0.0)
        cut = st.radio(
            "What is cut direction?",
            ('One Way Cut', 'Two Way Cut'),index=1,horizontal=True)
        cuff = st.radio(
            "What is cuff style?",
            ('Single Cuff', 'Double Cuff'),index=0,horizontal=True)
        sleeve = st.radio(
            "What is sleeve length?",
            ('Long Sleeve', 'Short Sleeve'),index=0,horizontal=True)
        avg_neck_size = st.slider('What is average neck size', min_value=28.0, max_value=40.0, value=32.0,
                                   step=0.2)
        marker_width = st.slider('What is marker width', min_value=48.0, max_value=80.0, value=60.0,
                                  step=0.2)
        plan_cut_qty = st.number_input('Plan Cut Qty',min_value=0)

        submitted = st.form_submit_button("Predict YY")

        if submitted:

            if pattern == "Solid":
                solid = 1
            else:
                solid = 0
            if pattern == "Stripe":
                stripe = 1
            else:
                stripe = 0
            if pattern == "Check":
                check = 1
            else:
                check = 0

            if cut == "One Way Cut":
                one_way_cut = 1
            else:
                one_way_cut =0
            if cut == "Two Way Cut":
                two_way_cut = 1
            else:
                two_way_cut =0

            if cut == "One Way Cut":
                one_way_cut = 1
            else:
                one_way_cut =0
            if cut == "Two Way Cut":
                two_way_cut = 1
            else:
                two_way_cut =0

            if cuff == "Single Cuff":
                single_cuff = 1
            else:
                single_cuff =0
            if cuff == "Double Cuff":
                double_cuff = 1
            else:
                double_cuff =0

            if sleeve == "Long Sleeve":
                long_sleeve = 1
            else:
                long_sleeve =0
            updated_date = date.today()

            predictor = [solid,stripe,check,one_way_cut,two_way_cut,long_sleeve,plan_cut_qty,repeat_x,repeat_y,avg_neck_size,double_cuff,marker_width]
            X_input = np.array(predictor).reshape(1,-1)

            # Change to display format
            X_display = pd.DataFrame(X_input,columns=["solid","stripe","check","one_way_cut","two_way_cut","long_sleeve","plan_cut_qty","repeat_x","repeat_y","avg_neck_size","double_cuff","marker_width"])
            X_display['solid']=X_display['solid'].astype(int)
            X_display['stripe'] = X_display['stripe'].astype(int)
            X_display['check'] = X_display['check'].astype(int)
            X_display['one_way_cut'] = X_display['one_way_cut'].astype(int)
            X_display['two_way_cut'] = X_display['two_way_cut'].astype(int)
            X_display['long_sleeve'] = X_display['long_sleeve'].astype(int)
            X_display['plan_cut_qty'] = X_display['plan_cut_qty'].astype(int)
            X_display['double_cuff'] = X_display['double_cuff'].astype(int)
            st.write(X_display)
            YY_estimation = pd.DataFrame(reg.predict(X_input),columns=["Predicted YY = "])
            predicted_yy = reg.predict(X_input)[0]
            YY_estimation_noindex = YY_estimation.to_string(index=False)
            st.write(YY_estimation_noindex)

            conn = sqlite3.connect('test.db')
            c = conn.cursor()
            # c.execute('''DROP TABLE QUOTATION''')
            c.execute('''CREATE TABLE if not exists QUOTATION
                        (updated_date text NOT NULL,
                        customer text NOT NULL,
                        solid Integer NOT NULL,
                        stripe Integer NOT NULL,
                        chk Integer Not NULL,
                        one_way_cut Integer NOT NULL,
                        two_way_cut Integer NOT NULL,
                        long_sleeve Integer NOT NULL,
                        plan_cut_qty Real NOT NULL,
                        repeat_x Real NOT NULL,
                        repeat_y Real NOT NULL,
                        avg_neck_size Real NOT NULL,
                        double_cuff Integer NOT NULL,
                        marker_width Real NOT NULL,
                        predicted_yy Real NOT NULL);''')


            # st.write(customer, solid, stripe, check, one_way_cut, two_way_cut, long_sleeve, plan_cut_qty, repeat_x, repeat_y,
            # avg_neck_size, double_cuff, marker_width, predicted_yy)

            c.execute("INSERT INTO QUOTATION (updated_date,customer,solid,stripe,chk,one_way_cut,two_way_cut,long_sleeve,plan_cut_qty,repeat_x,repeat_y,avg_neck_size,double_cuff,marker_width,predicted_yy) \
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (updated_date,customer, solid, stripe, check, one_way_cut, two_way_cut, long_sleeve, plan_cut_qty, repeat_x, repeat_y,
             avg_neck_size, double_cuff, marker_width,predicted_yy))
            conn.commit()

            statement = '''SELECT * FROM QUOTATION'''
            c.execute(statement)
            db = pd.read_sql_query(statement,conn)
            AgGrid(db)
            print("All the data")
            output = c.fetchall()
            for row in output:
                st.write(row)

            conn.commit()
            conn.close()


    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    statement_1 = '''SELECT * FROM QUOTATION'''
    c.execute(statement_1)
    conn.commit()
    db_1 = pd.read_sql_query(statement_1, conn)
    csv = convert_df(db_1)
    conn.close()
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='db_1.csv',
        mime='text/csv',
        )
