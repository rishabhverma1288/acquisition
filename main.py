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
import base64

st.title("New Users App")

st.write("""# Machine learning through Streamlit """)
data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])


if data is None:
    st.write("Select a CSV or n XLSX File")

else:
    @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def reader(dat):
        try:
            r = dat.name
            datasetname = st.write(dat.name)
            if str(r).endswith('csv'):
                qw = pd.read_csv(dat)

            elif str(r).endswith('xlsx'):
                qw = pd.read_excel(dat)
            return qw
        except:
            st.write("Select a CSV or n XLSX File")

    df = reader(data)

try:
    st.write(df)
except:
    st.write("")

@st.cache(suppress_st_warning=True)
def plam():
    df['Customer_Email'] = df['Customer_Email'].str.lower()
    df['Order_date']= pd.to_datetime(df['Order_date'])
    df['Day'] = df['Order_date'].dt.day
    df['Month'] = df['Order_date'].dt.month
    df['Year'] = df['Order_date'].dt.year
    df['Date'] = df['Order_date'].dt.date
    df['Time'] = df['Order_date'].dt.time
    df.sort_values('Order_date',inplace=True)
    df['Day']= df['Day'].apply(str)
    df['Month']= df['Month'].apply(str)
    df['Year']= df['Year'].apply(str)
    df['Date'] = df['Date'].apply(str)
    df['Time']= df['Time'].apply(str)
    df['Mon_Year'] = df['Month'].str.cat(df['Year'],sep='/')
    df['Period'] = df.Customer_Email.str.cat(df[['Date','Time']],sep='/')
    df3 = df.drop_duplicates('Customer_Email',keep='first')
    df4 = df3['Period']
    df4 = pd.DataFrame(df4)
    r = 1
    df4['Yes'] = r
    df_main = pd.merge(df, df4, on ='Period', how ='left')
    df_main['Yes'] = df_main['Yes'].fillna(0)
    def user_new(x):
        if x == 1:
            return 'New'
        else:
            return 'Old'
    df_main['New User'] = df_main['Yes'].apply(user_new)
    df_main['Sum'] = r

    def month_names(x):
        return x.strftime("%B")
    df_main['Month_Name'] = df_main['Order_date'].apply(month_names)

    def week_number_of_month(date_value):
         return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)
    df_main['Month Week Number'] = df_main['Order_date'].apply(week_number_of_month)

    def month_part(x):
        if int(x) > 22:
            return 'Month P4'
        elif int(x) > 14:
            return 'Month P3'
        elif int(x) > 7:
            return 'Month P2'
        else:
            return 'Month P1'
    df_main['Month Parts'] = df_main['Day'].apply(month_part)
    df_main['Month Parts Month Wise'] = df_main.Month_Name.str.cat(df_main['Month Parts'],sep=' - ')
    def hours(x):
        return int(x.hour)
    df_main['Hours'] = df_main['Order_date'].apply(hours)
    def hour_part(x):
        if x >= 18:
            return 'Night'
        elif x >= 15:
            return 'Evening'
        elif x >= 12:
            return 'Afternoon'
        elif x >= 6:
            return 'Morning'
        elif x >= 0:
            return 'Midnight'
    df_main['Hours_Part'] = df_main['Hours'].apply(hour_part)

    return df_main

if st.checkbox('Show Transform'):
    p = plam()
    st.write(p)
    csv = p.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)
