#!/usr/bin/env python
# coding: utf-8

#Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

#dataframe.describe()
#dataframe.nunique()
#dataframe.info()

#Insert picture
st.image("https://q46g328tbx259g0tu3qkm19a-wpengine.netdna-ssl.com/wp-content/uploads/2018/05/mc-cateory-header-image-data-analytics-min.jpg")
#Insert title
st.title("Marketing Analytics for XYZ Company")

select_box = st.selectbox('Navigation', ['Exploratory Analysis', 'Customer Segmentation'])

if select_box == 'Exploratory Analysis':

#Load the data
    url = "https://drive.google.com/file/d/1Q-8zU0VLjf5bPMbIL2ekTwOk41CQf1C_/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    dataframe = pd.read_csv(path)

    #dataframe = pd.read_csv("C:/Users/HP/Downloads/streamlit/marketing_data.csv")
    shape = dataframe.shape

#Sidebar
    if st.sidebar.checkbox("dataset shape"):
        st.sidebar.write(shape)

    rows = st.sidebar.slider("How many rows to display:", 5, 10, 20)

    st.sidebar.write("Filters")

#Filters
    campaign_options = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    campaign_choice =  st.sidebar.multiselect('Select the campaigns that you are interested in:', campaign_options)

    #channel_options = ['Deal','Web','Catalog','Store']
    #channel_choice =  st.sidebar.multiselect('Select the channels that you are interested in:', channel_options)


    food_options = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    food_choice =  st.sidebar.multiselect('Select the food categories that you are interested in:', food_options)

#education_options = dataframe['Education'].drop_duplicates()
#education_choice = st.sidebar.multiselect('Select the education categories that you are interested in:', education_options)

#maritalstatus_options = dataframe["Marital_Status"].drop_duplicates()
#maritalstatus_choice = st.sidebar.multiselect("Select the marital status that you are interested in:", maritalstatus_options)

#country_options = dataframe["Country"].drop_duplicates()
#country_choice =  st.sidebar.multiselect("Select the countries that you are interested in:", country_options)

#Display the dataset
    if st.button("Load dataset"):
        col, col2 = st.beta_columns(2)
        col.write(dataframe.head(rows))

#Campaign performance
    st.header('Campaign performance')

#Remove white space from columns names
    dataframe.columns = dataframe.columns.str.replace(' ', '')

#Remove $ from Income column
    dataframe['Income'] = dataframe['Income'].str.replace('$', '')

#Remove "," from Income column
    dataframe['Income'] = dataframe['Income'].str.replace(',', '')

#Convert string to float
    dataframe['Income'] = dataframe['Income'].astype('float')

#Fill NA values with median
    dataframe['Income'] = dataframe['Income'].fillna(dataframe['Income'].median())

#Convert to datetime
    dataframe['Dt_Customer'] = pd.to_datetime(dataframe['Dt_Customer'])

#Calculate age
    dataframe['Customer_Age'] = dataframe['Dt_Customer'].dt.year - dataframe['Year_Birth']

#calculate success rate (percent accepted)
    accepted_campaigns = pd.DataFrame(dataframe[campaign_choice].mean()*100, columns=['Accepted (%)']).reset_index()

# Total Amount Spent
    mnt_cols = [col for col in dataframe.columns if 'Mnt' in col]
    dataframe['TotalMnt'] = dataframe[mnt_cols].sum(axis=1)

# Total Purchases
    purchases_cols = [col for col in dataframe.columns if 'Purchases' in col]
    dataframe['TotalPurchases'] = dataframe[purchases_cols].sum(axis=1)

# list of columns for channels
    channel_cols = [col for col in dataframe.columns if 'Num' and 'Purchases' in col]

#Plots
    import plotly.express as px

#Plot histogram
    fig1 = px.bar(accepted_campaigns.sort_values('Accepted (%)'), x = 'Accepted (%)', y = "index",
             title="Marketing campaign success rate", color_discrete_sequence =['#1167b1'])
    fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Plot histogram
    fig2 = px.histogram(dataframe, x = 'Education' , y = 'AcceptedCmp4', color_discrete_sequence =['#1167b1'], title="Acceptance of campaign 4 across customers").update_xaxes(categoryorder = 'total descending')
    fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    col1, col2 = st.beta_columns(2)
    col1.write(fig1)
    col2.write(fig2)

#Plot histogram
    fig5 = px.histogram(dataframe, x = 'Country', y = "TotalPurchases", color_discrete_sequence =['#1167b1'],
                    title = 'Total Number of Purchases by Country').update_xaxes(categoryorder = 'total descending')
    fig5.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Plot pie chart
    Purchases = dataframe.loc[:,'NumDealsPurchases':'NumWebVisitsMonth']
    total_purchase_each = np.sum(Purchases.iloc[:,:-1], axis=0)
    labels = ['Deal','Web','Catalog','Store']
    data = total_purchase_each
    fig6 = px.pie(dataframe, values= data , names = labels, title="% Purchases across channels", color = labels , color_discrete_map={'Deal':'#1167b1',
                                 'Web':'lightcyan',
                                 'Catalog':'royalblue',
                                 'Store':'darkblue'})
    fig6.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Country and Channel Performance
    st.header('Country and Channel performance')

    col5, col6 = st.beta_columns(2)
    col5.write(fig5)
    col6.write(fig6)

#Plot histogram
    fig7 = px.histogram(dataframe, x = 'Education', y = "MntWines", title="Amount of wine purchased across customers", color_discrete_sequence =['#1167b1']).update_xaxes(categoryorder = 'total descending')
    fig7.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Plot histogram
    fig8 = px.histogram(dataframe, x = 'Education', y = "MntFishProducts", title="Amount of fish products purchased across customers", color_discrete_sequence =['#1167b1']).update_xaxes(categoryorder = 'total descending')
    fig8.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

# calculate average amount spent
    product_success = pd.DataFrame(dataframe[food_choice].mean(), columns=['Amount']).reset_index()
#Plot barchart
    fig11 = px.bar(product_success.sort_values('Amount'), x = 'Amount', y = "index", title='Average amount spent', color_discrete_sequence =['#1167b1'])
    fig11.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Plot histogram
    fig12 = px.histogram(dataframe, x = 'Marital_Status', y = "TotalPurchases", title="Total Purchases across customers", color_discrete_sequence =['#1167b1']).update_xaxes(categoryorder = 'total descending')
    fig12.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Customer Profiles
    st.header('Customer profiles')

    col11, col12 = st.beta_columns(2)
    col11.write(fig11)
    col12.write(fig12)

#Scatter plot
    fig3 = px.scatter(dataframe[dataframe['Income'] < 200000], x="Income", y="TotalMnt",  title="Income vs. Total Amount Purchased", color_discrete_sequence =['#1167b1'], trendline="ols")
    fig3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

#Scatter plot
    fig4 = px.scatter(dataframe, x = 'TotalPurchases', y = "MntGoldProds", title = "Amount of Gold purchased vs. Total Amount Purchased", color_discrete_sequence =['#1167b1'], trendline="ols")
    fig4.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    col3, col4 = st.beta_columns(2)
    col3.write(fig3)
    col4.write(fig4)

#Product Preferences
    st.header('Product preferences')

    col7, col8 = st.beta_columns(2)
    col7.write(fig7)
    col8.write(fig8)

if select_box == 'Customer Segmentation':
#Customer segmentation (machine learning part)

#Load the data
    url = "https://drive.google.com/file/d/1Q-8zU0VLjf5bPMbIL2ekTwOk41CQf1C_/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    dataframe = pd.read_csv(path)

#load the libraries
    import datetime
    import plotly.express as px
    from datetime import date
    import matplotlib
    import seaborn as sns
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn import metrics
    from sklearn.mixture import GaussianMixture
    import warnings
    warnings.filterwarnings('ignore')

    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    data_folder = "/kaggle/input/arketing-campaign/"

#Remove white space from columns names
    dataframe.columns = dataframe.columns.str.replace(' ', '')

#Remove $ from Income column
    dataframe['Income'] = dataframe['Income'].str.replace('$', '')

#Remove "," from Income column
    dataframe['Income'] = dataframe['Income'].str.replace(',', '')

#Convert string to float
    dataframe['Income'] = dataframe['Income'].astype('float')

#Fill NA values with median
    dataframe['Income'] = dataframe['Income'].fillna(dataframe['Income'].median())

#Spending variable creation
    dataframe['Spending']=dataframe['MntWines']+dataframe['MntFruits']+dataframe['MntMeatProducts']+dataframe['MntFishProducts']+dataframe['MntSweetProducts']+dataframe['MntGoldProds']

#Seniority variable creation
    last_date = date(2014, 6, 29)
    dataframe['Seniority']= pd.to_datetime(dataframe['Dt_Customer'], dayfirst=True,format = '%m/%d/%y')
    dataframe['Seniority'] = pd.to_numeric(dataframe['Seniority'].dt.date.apply(lambda x: (last_date - x)).dt.days, downcast='integer')/30

    dataset=dataframe[['Income','Spending','Seniority']]
    dataset.head()

#Scale the data
    scaler=StandardScaler()
    dataset=dataset[['Income','Seniority','Spending']]

    X_std=scaler.fit_transform(dataset)
    X = normalize(X_std,norm='l2')

#Calculate silhouette score and David Bouldin Score
    Covariance=['full','tied','diag','spherical']
    number_clusters=np.arange(1,21)
    results_=pd.DataFrame(columns=['Covariance type','Number of Clusters','Silhouette Score','Davies Bouldin Score'])
    for i in Covariance:
        for n in number_clusters:
            gmm_cluster=GaussianMixture(n_components=n,covariance_type=i,random_state=5)
            clusters=gmm_cluster.fit_predict(X)
            if len(np.unique(clusters))>=2:
                results_=results_.append({"Covariance type":i,'Number of Clusters':n,"Silhouette Score":metrics.silhouette_score(X,clusters),'Davies Bouldin Score':metrics.davies_bouldin_score(X,clusters)},ignore_index=True)
#results_.sort_values(by=["Silhouette Score"], ascending=False)[:10]

#Gaussian model
    number_clusters = np.arange(1, 10)
    models = [GaussianMixture(n, covariance_type='spherical',max_iter=2000, random_state=5).fit(X) for n in number_clusters]

#Plot number of clusters vs. aic
    fig13 = px.line(dataframe, x=number_clusters, y=[m.aic(X) for m in models], title="AIC vs. number of clusters", color_discrete_sequence =['#1167b1'], labels={'x':'number of clusters', 'y':'AIC'})
    fig13.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    gmm=GaussianMixture(n_components=4, covariance_type='spherical',max_iter=2000, random_state=5).fit(X)
    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis');

    proba = gmm.predict_proba(X)
    print(proba[:].round(2))

    dataset['Cluster'] = labels

    Probability=pd.DataFrame(proba.max(axis=1))
    dataset = dataset.reset_index().merge(Probability, left_index=True, right_index=True)
    dataset=dataset.rename(columns={0: "Probability"}).drop(columns=['index'])

    pd.options.display.float_format = "{:.0f}".format
    summary=dataset[['Income','Spending','Seniority','Cluster']]
    summary.set_index("Cluster", inplace = True)
    summary=summary.groupby('Cluster').describe().transpose()

#Rename clusters
    dataset=dataset.replace({0:'Stars',1:'Need attention',2:'High potential',3:'Leaky bucket'})

#Plot customer segmentation
    PLOT = go.Figure()
    for C in list(dataset.Cluster.unique()):


        PLOT.add_trace(go.Scatter3d(x = dataset[dataset.Cluster == C]['Income'],
                                y = dataset[dataset.Cluster == C]['Seniority'],
                                z = dataset[dataset.Cluster == C]['Spending'],
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(C)))
    PLOT.update_traces(hovertemplate='Income: %{x} <br>Seniority: %{y} <br>Spending: %{z}')


    PLOT.update_layout(width = 850, height = 850, title="Customer Segmentation", showlegend = True,
                   scene = dict(xaxis=dict(title = 'Income'),
                                yaxis=dict(title = 'Seniority'),
                                zaxis=dict(title = 'Spending')),
                   font = dict(family = "Gilroy", size = 15))

    PLOT.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="right", x=0.01))

    st.header("Customer segmentation")

    col13, col14 = st.beta_columns(2)
    col13.write(fig13)
    number = col14.slider("How many rows to display:", 10, 30, 2239)
    col14.write(dataset.head(number))

    col15, col16 = st.beta_columns(2)
    col15.write(PLOT)
    col16.write(" ")
    col16.write(" ")
    col16.write(" ")
    col16.write(" ")
    col16.write(" ")
    col16.write(" ")
    col16.write("Star cluster is composed of old customers with high income and high spending amount")
    col16.write("Need attention cluster is composed of new customers with below average income and small spending amount")
    col16.write("High potential cluster is composed of new customers with high income and high spending amount")
    col16.write("Leaky bucket cluster is composed of old customers with below average income and small spending amount")

    st.info("Conclusion:")
    st.write("•	The most successful advertising campaign was the 4th campaign, so the suggested action is to conduct future advertising campaigns using similar strategy.")
    st.write("•	The most successful products are wines and meats, so the suggested action is to focus advertising campaigns on boosting sales of the less popular items. (fruits, sweet products…) ")
    st.write("•	The underperforming channels are deals and stores purchases and the best performing channels are web and catalog purchases, so the suggested action is to focus advertising campaigns on the more successful channels to reach more customers.")
    st.write("•	Some customers with low income are spending a lot, which means we could try to apply a marketing strategy initially defined for Stars customers to them.")
