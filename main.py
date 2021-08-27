import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt



st.title("Streamlit App")

st.write("""# Machine learning through Streamlit """)
data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])

if data is None:
    st.write("Select a CSV or n XLSX File")

else:

    def reader(dat):
        try:
            r = dat.name
            datasetname = st.write(dat.name)
            if str(r).endswith('csv'):
                reader.m = pd.read_csv(dat)

            elif str(r).endswith('xlsx'):
                reader.m = pd.read_excel(dat)

        except:
            st.write("Select a CSV or n XLSX File")
    reader(data)
    state_dropdown = st.sidebar.multiselect('Charts',['Pairplot','Barplot','Jointplot'])
    target_column = st.sidebar.selectbox('Main Target',tuple(reader.m.columns))

    dropper = st.sidebar.checkbox('Drop Columns?')
    if dropper:
        drop_column = st.sidebar.multiselect('Columns',list(reader.m.columns))
        if st.sidebar.checkbox('Drop'):
            reader.m = reader.m.drop(drop_column,axis=1)

    dumy = st.sidebar.checkbox('Make Dummies?')
    if dumy:
        dumy_columns = st.sidebar.multiselect('Columns',list(reader.m.columns),key = '12')
        if len(dumy_columns) == 1:
            if st.sidebar.checkbox('Dummy Execute'):
                dummies = pd.get_dummies(reader.m[dumy_columns[0]],drop_first=True)
                reader.m = reader.m.drop([dumy_columns[0]],axis=1)
                reader.m = pd.concat([reader.m,dummies],axis=1)
        else:
            if st.sidebar.checkbox('Dummy Execute'):
                dummies = pd.get_dummies(reader.m[dumy_columns],drop_first=True)
                reader.m = reader.m.drop(dumy_columns,axis=1)
                reader.m = pd.concat([reader.m,dummies],axis=1)


    st.write(reader.m)

    if st.sidebar.checkbox('Show Correlation plot'):
        corr_col = st.sidebar.selectbox('Main Target',tuple(reader.m.columns),key='11')
        st.write(reader.m.corr()[corr_col])


try:
    main_df = reader.m
except:
    print('hi')

@st.cache(suppress_st_warning=True)
def pair(ft):
    for x  in state_dropdown:
        if x == 'Pairplot':
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.pairplot(reader.m))
            st.pyplot()
        elif x == 'Barplot':
            try:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                x_g = st.selectbox("Barplot X Column",tuple(reader.m.columns))
                y_g = st.selectbox("Barplot Y Column",tuple(reader.m.columns))
                st.write(sns.barplot(x=x_g, y=y_g,data=reader.m))
                st.pyplot()
            except:
                st.write('Please Select Valid Column')
        elif x == 'Jointplot':
            x_g = st.selectbox("Barplot X Column",tuple(reader.m.columns),key='1')
            y_g = st.selectbox("Barplot Y Column",tuple(reader.m.columns),key='2')
            kin = st.selectbox("Select kind",("scatter","reg","hex"))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            if kin == 'scatter':
                try:
                    target = st.selectbox("Target Column",tuple(reader.m.columns))
                    st.write(sns.jointplot(x=x_g,y=y_g,hue=target,data=reader.m,kind=kin))
                    st.pyplot()
                except:
                    st.write('Please Choose a Valid Target Column')
            else:
                st.write(sns.jointplot(x=x_g,y=y_g,data=reader.m,kind=kin))
                st.pyplot()

if data is not None:
    pair(state_dropdown)
    classifier = st.sidebar.selectbox("Select Classifier",("None",'Linear Regression',"Logistic","Decision Tree","SVM","Random Forest"))

@st.cache(suppress_st_warning=True)
def split_step1():
    if classifier == 'Logistic':
        X_cols = st.sidebar.multiselect('X Columns',list(reader.m.columns),default=list(reader.m.columns))
        y_cols = st.sidebar.selectbox('Target Column',tuple(reader.m.columns),key='4')
        split_step1.X = reader.m[X_cols]
        split_step1.y = reader.m[y_cols]
        split_step1.splt = st.sidebar.slider('Split By',min_value =0.0,max_value = 1.0,value=0.25)
    else:
        X_cols = st.sidebar.multiselect('X Columns',list(reader.m.columns),default=list(reader.m.columns))
        y_cols = st.sidebar.selectbox('Target Column',tuple(reader.m.columns),key = '3')
        split_step1.X = reader.m[X_cols].values
        split_step1.y = reader.m[y_cols].values
        split_step1.splt = st.sidebar.slider('Split By',min_value =0.0,max_value = 1.0,value=0.25)


if data is not None:
    splitter = st.sidebar.checkbox('Split')




def decision():
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    st.write(classification_report(y_test,predictions))
    st.write(confusion_matrix(y_test,predictions))

    if st.sidebar.button('Export Predictions'):
        predictions = dtree.predict(split_step1.X)
        main_df['predictions'] = predictions
        main_df.to_excel('predicted_decision_tree.xlsx',index=0)
    to_be_predicted = st.sidebar.file_uploader("Upload a Dataset", type=["csv", "xlsx"],key='5')
    if to_be_predicted is not None:
        def p_reader(dat):
            try:
                r = dat.name
                datasetname = st.write(dat.name)
                if str(r).endswith('csv'):
                    p_main_df = pd.read_csv(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
                elif str(r).endswith('xlsx'):
                    p_main_df = pd.read_excel(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
            except:
                st.sidebar.write("Select a CSV or n XLSX File")
        p_reader(to_be_predicted)

        splitter1 = st.sidebar.checkbox('Split New Dataset')
        if splitter1:
            X_cols_pre = st.sidebar.multiselect('X Columns',list(p_main_df.columns))
            Xp = p_main_df[X_cols_pre].values
        if st.sidebar.button('Do Predictions'):
            predictions1 = dtree.predict(Xp)
            p_main_df['predictions'] = predictions1
            p_main_df.to_excel('predicted_new_dataset_decis.xlsx',index=0)

    #elif rp == 'Logistic':

def logmo():
    log_model = LogisticRegression()
    log_model = log_model.fit(X_train,y_train)
    predictions = log_model.predict(X_test)
    st.write(classification_report(y_test,predictions))
    st.write(confusion_matrix(y_test,predictions))

    if st.sidebar.button('Export Predictions'):
        predictions = log_model.predict(split_step1.X)
        main_df['predictions'] = predictions
        main_df.to_excel('predicted_logistic.xlsx',index=0)

    to_be_predicted = st.sidebar.file_uploader("Upload a Dataset", type=["csv", "xlsx"],key='7')
    if to_be_predicted is not None:
        def p_reader(dat):
            try:
                r = dat.name
                datasetname = st.write(dat.name)
                if str(r).endswith('csv'):
                    p_main_df = pd.read_csv(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
                elif str(r).endswith('xlsx'):
                    p_main_df = pd.read_excel(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
            except:
                st.sidebar.write("Select a CSV or n XLSX File")
        p_reader(to_be_predicted)

        splitter1 = st.sidebar.checkbox('Split New Dataset')
        if splitter1:
            X_cols_pre = st.sidebar.multiselect('X Columns',list(p_main_df.columns))
            Xp = p_main_df[X_cols_pre].values
        if st.sidebar.button('Do Predictions'):
            predictions1 = log_model.predict(Xp)
            p_main_df['predictions'] = predictions1
            p_main_df.to_excel('predicted_new_dataset_log.xlsx',index=0)
#else:
    #st.write('Upload dataset on which model will do predictions')

def svc():
    svc_model = SVC()
    svc_model = svc_model.fit(X_train,y_train)
    predictions = svc_model.predict(X_test)
    st.write(classification_report(y_test,predictions))
    st.write(confusion_matrix(y_test,predictions))

    if st.sidebar.button('Export Predictions'):
        predictions = svc_model.predict(split_step1.X)
        main_df['predictions'] = predictions
        main_df.to_excel('predicted_SVM.xlsx',index=0)

    to_be_predicted = st.sidebar.file_uploader("Upload a Dataset", type=["csv", "xlsx"],key='9')
    if to_be_predicted is not None:
        def p_reader(dat):
            try:
                r = dat.name
                datasetname = st.write(dat.name)
                if str(r).endswith('csv'):
                    p_main_df = pd.read_csv(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
                elif str(r).endswith('xlsx'):
                    p_main_df = pd.read_excel(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
            except:
                st.sidebar.write("Select a CSV or n XLSX File")
        p_reader(to_be_predicted)

        splitter1 = st.sidebar.checkbox('Split New Dataset')
        if splitter1:
            X_cols_pre = st.sidebar.multiselect('X Columns',list(p_main_df.columns))
            Xp = p_main_df[X_cols_pre].values
        if st.sidebar.button('Do Predictions'):
            predictions1 = log_model.predict(Xp)
            p_main_df['predictions'] = predictions1
            p_main_df.to_excel('predicted_new_dataset_svm.xlsx',index=0)

def lmod():
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    
    predictions = lm.predict( X_test)
    if st.checkbox('Show Prediction Plot'):
        st.write(plt.scatter(y_test,predictions))
        st.write(sns.jointplot(x=y_test,y=predictions,kind='reg'))
        st.pyplot()
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    st.write('MAE:', metrics.mean_absolute_error(y_test, predictions))
    st.write('MSE:', metrics.mean_squared_error(y_test, predictions))
    st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    sns.distplot((y_test-predictions),bins=1)

    if st.sidebar.button('Export Predictions'):
        predictions = lm.predict(split_step1.X)
        main_df['predictions'] = predictions
        main_df.to_excel('predicted_linear.xlsx',index=0)

    to_be_predicted = st.sidebar.file_uploader("Upload a Dataset", type=["csv", "xlsx"],key='9')
    if to_be_predicted is not None:
        def p_reader(dat):
            try:
                r = dat.name
                datasetname = st.write(dat.name)
                if str(r).endswith('csv'):
                    p_main_df = pd.read_csv(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
                elif str(r).endswith('xlsx'):
                    p_main_df = pd.read_excel(dat)
                    st.write('Dataset to be predicted yet')
                    st.write(p_main_df)
            except:
                st.sidebar.write("Select a CSV or n XLSX File")
        p_reader(to_be_predicted)

        splitter1 = st.sidebar.checkbox('Split New Dataset')
        if splitter1:
            X_cols_pre = st.sidebar.multiselect('X Columns',list(p_main_df.columns))
            Xp = p_main_df[X_cols_pre].values
        if st.sidebar.button('Do Predictions'):
            predictions1 = log_model.predict(Xp)
            p_main_df['predictions'] = predictions1
            p_main_df.to_excel('predicted_new_dataset_svm.xlsx',index=0)


if data is not None:
    if splitter:
        split_step1()
        try:
            X_train,X_test, y_train, y_test = train_test_split(split_step1.X,split_step1.y,test_size=split_step1.splt,
                                                            random_state=101)
        except:
            st.write('Please Select Split Option in Sidebar and a Categorical Binary column in Target Column Selection')

        executioner = st.sidebar.checkbox('Execute Classifier')
        if executioner:
            if classifier == 'Decision Tree':
                decision()
            elif classifier == 'Logistic' :
                logmo()
            elif classifier == 'SVM' :
                svc()
            elif classifier == 'Linear Regression' :
                lmod()
