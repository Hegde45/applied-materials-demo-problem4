import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

st.set_page_config(layout="wide", page_title='Problem 4')

@st.cache_data(experimental_allow_widgets=True)
def load_csv():
    csv = pd.read_csv("train.csv")
    if 'Id' in csv:
        del csv['Id']
    return csv

if 'train_data' not in st.session_state:
    st.session_state['train_data'] = pd.DataFrame(load_csv())
    for column in st.session_state['train_data'].columns:
        if st.session_state['train_data'][column].dtype == 'object': 
            st.session_state['train_data'][column]=st.session_state['train_data'][column].astype('category').cat.codes

# Calculate summary statistics
@st.cache_data(experimental_allow_widgets=True)
def summary_stats():
    st.subheader('Summary Statistics', anchor=False)
    stats = st.session_state['train_data'].describe()
    st.write(stats)

# Display the correlation
@st.cache_data(experimental_allow_widgets=True)
def heatmap_all_correlations():

    df = st.session_state['train_data'].copy()

    correlation_matrix = df.corr()
    plt.figure(figsize=(60, 40))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, cbar=1, square=1, annot_kws={'size': 15})
    plt.tick_params(axis = 'x', labelsize = 26)
    plt.tick_params(axis = 'y', labelsize = 26)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)

@st.cache_data(experimental_allow_widgets=True)
def heatmap_high_correlations():
    df = st.session_state['train_data'].copy()
    for column in df.columns:
        if df[column].dtype == 'object': 
            df[column]=df[column].astype('category').cat.codes
            
    correlation_matrix = df.corr()
    threshold = 0.5
    filtered_corr_df = correlation_matrix[((correlation_matrix >= threshold) | (correlation_matrix <= -threshold)) & (correlation_matrix != 1.000)]

    plt.figure(figsize=(60, 40))
    sns.heatmap(filtered_corr_df, annot=True, cmap='Blues', xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, cbar=1, square=1, annot_kws={'size': 15})
    plt.tick_params(axis = 'x', labelsize = 18)
    plt.tick_params(axis = 'y', labelsize = 18)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    st.pyplot(plt)

# Display the scatter plots
@st.cache_data(experimental_allow_widgets=True)
def scatter_plot(selected_features):
    for feature in selected_features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feature, y='SalePrice', data=st.session_state['train_data'])
        st.pyplot(plt)

@st.cache_data(experimental_allow_widgets=True)
def scatter_matrix_plot(columns, serial_no):
    name = 'Scatter matrix plot - Category ' + str(serial_no)
    st.subheader(name, anchor=False)
    subset_df = st.session_state['train_data'][columns]
    sns.set(style="ticks")
    sns.pairplot(subset_df)
    st.pyplot(plt)

@st.cache_data(experimental_allow_widgets=True)
def default_dataframes():
    st.title('''**Problem 4 - Housing Prices Analysis**''', anchor=False)
    st.write("**`PLEASE NOTE : Charts are blurred to manage the resources, as its free deploy with low limits`**")
    
    st.subheader('Dataframe', anchor=False)
    st.dataframe(st.session_state['train_data'], use_container_width=True)

@st.cache_data(experimental_allow_widgets=True)
def heatmap_wrapper():
    correlation_option_selected = st.radio(
        "Please select an option for correlations",
        ('All Correlations',
        'High Correlations'))
    if correlation_option_selected == 'All Correlations':
        st.subheader('All Correlations', anchor=False)
        heatmap_all_correlations()
    elif correlation_option_selected == 'High Correlations':
        st.subheader('High Correlations', anchor=False)
        heatmap_high_correlations()

@st.cache_data(experimental_allow_widgets=True)
def scatter_plot_wrapper():
    st.subheader('Scatter plot', anchor=False)
    selected_features = st.multiselect('Select features to plot against sale price', st.session_state['train_data'].columns)
    scatter_plot(selected_features)

@st.cache_data(experimental_allow_widgets=True)
def numeric_stats():
    st.subheader('summary numeric', anchor=False)
    numeric_cols = st.session_state['train_data'].select_dtypes(include=['int64', 'float64'])
    summary_stats2 = numeric_cols.describe()
    st.dataframe(summary_stats2)

@st.cache_data(experimental_allow_widgets=True)
def categorical_stats():
    st.subheader('summary categorical', anchor=False)
    categorical_cols = st.session_state['train_data'].select_dtypes(include=['int8'])
    summary_stats3 = categorical_cols.describe()
    st.dataframe(summary_stats3)

@st.cache_data(experimental_allow_widgets=True)
def scatter_matrix_plot_wrapper():
    scatter_matrix_plot_option_1 = 'MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood'
    scatter_matrix_plot_option_2 = 'Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd'
    scatter_matrix_plot_option_3 = 'RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond'
    scatter_matrix_plot_option_4 = 'Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF'
    scatter_matrix_plot_option_5 = 'Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea, BsmtFullBath,BsmtHalfBath, FullBath,HalfBath,BedroomAbvGr'
    scatter_matrix_plot_option_6 = 'KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu'
    scatter_matrix_plot_option_7 = 'GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF'
    scatter_matrix_plot_option_8 = 'OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition'

    scatter_matrix_plot_option_selected = st.radio(
        "Please select an option for scatter matrix plot, based on the variable groups",
        (scatter_matrix_plot_option_1,
        scatter_matrix_plot_option_2,
        scatter_matrix_plot_option_3,
        scatter_matrix_plot_option_4,
        scatter_matrix_plot_option_5,
        scatter_matrix_plot_option_6,
        scatter_matrix_plot_option_7,
        scatter_matrix_plot_option_8))

    if scatter_matrix_plot_option_selected == scatter_matrix_plot_option_1:
        scatter_matrix_plot(scatter_matrix_plot_option_1.split(","), 1)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_2:
            scatter_matrix_plot(scatter_matrix_plot_option_2.split(","), 2)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_3:
            scatter_matrix_plot(scatter_matrix_plot_option_3.split(","), 3)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_4:
            scatter_matrix_plot(scatter_matrix_plot_option_4.split(","), 4)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_5:
            scatter_matrix_plot(scatter_matrix_plot_option_5.split(","), 5)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_6:
            scatter_matrix_plot(scatter_matrix_plot_option_6.split(","), 6)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_7:
            scatter_matrix_plot(scatter_matrix_plot_option_7.split(","), 7)
    elif scatter_matrix_plot_option_selected == scatter_matrix_plot_option_8:
            scatter_matrix_plot(scatter_matrix_plot_option_8.split(","), 8)

def app():
    default_dataframes()
    st.divider()
    summary_stats()
    st.divider()
    heatmap_wrapper()
    st.divider()
    scatter_plot_wrapper()
    st.divider()
    numeric_stats()
    st.divider()
    categorical_stats()
    st.divider()
    scatter_matrix_plot_wrapper()

if __name__ == '__main__':
    app()