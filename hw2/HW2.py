#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:02:34 2022

@author: humeyraakyuz
"""


import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import  scipy.signal.signaltools
from sklearn import preprocessing
import recommender_sys
import reco_sys_with_reg

def root_mean_squared_error(actual, predictions):
    return np.sqrt(mean_squared_error(actual, predictions))


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered


if __name__ == '__main__':

    st.header("Building a Recommender System ")
    st.markdown("""
     In that recommender system, we have 2 list: b_user and b_item, correspond to 
     bias for each user and item respectively.
    
     For Example when we want to predict that how user i 
     rates item j, we basically sum the bias of the corresponding user and item
     
     """)
    
    st.subheader("Formulating the Model")
    st.markdown("#### General Model")
    st.latex(r"\hat{r_{i,j}}=\beta_i^{user} + \beta_j^{item}")
    
    st.markdown("#### Loss Function")
    
    
    st.latex(
        r"L(\beta^{user},\beta^{item})=\sum_{{i,j}\isin{S}}{(r_{i,j}- \hat{r_{i,j}})^2)/{2} }")
    
    st.latex(
        r"L(\beta^{user},\beta^{item})=\sum_{{i,j}\isin{S}}{(r_{i,j}- \beta_i^{user} - \beta_j^{item})^2)/{2} }")
    
    
    st.markdown("#### Partial derivatives")
    
    st.markdown("""
       Here we calculate partial derivative for each user and item. We  should be careful about indexes.
       Because for example when we want to calculate partial derivative for user i, While calculating
       error, we can only take acoount the item that user i has rated. Therefore we should be carefull and
       use S set carefully.
                """)
    
    st.latex(
        r"\forall{i} \qquad \qquad \frac{\partial L(\beta^{user},\beta^{item})}{\partial \beta_i^{user}} = - \sum_{{i,j}\isin{S}}{(r_{i,j}- \beta_i^{user} - \beta_j^{item})}")
                
    
    st.latex(
        r"\forall{j} \qquad \qquad \frac{\partial L(\beta^{user},\beta^{item})}{\partial \beta_j^{item}} = - \sum_{{i,j}\isin{S}}{(r_{i,j}- \beta_i^{user} - \beta_j^{item})}")
                
    st.header("Dataset")
    st.subheader("Original Dataset")
    
    
    df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    R = df.pivot(index='user_id', columns='item_id', values='rating').values
    st.dataframe(R)
    
    
    st.header("Model 1(Without Regularization)") 
    
    #separating train and test set 
    irow, jcol = np.where(~np.isnan(R))
    idx = np.random.choice(np.arange(100000), 20000, replace=False)
    test_irow = irow[idx]
    test_jcol = jcol[idx]
    
    R_copy = R.copy()
    
    for i, j in zip(test_irow, test_jcol):
            R_copy[i][j] = np.nan
       
    train_irow, train_jcol = np.where(~np.isnan(R_copy))
        
    r = R_copy.copy()
    
    st.subheader("Train set")
    st.dataframe(r)
    
    #model fit 
    
    reco_model = recommender_sys.recoSys( alpha = 0.02, iteration  = 500)
    reco_model.fit(r)
    y_pred, y= reco_model.predict1(R, test_irow, test_jcol)
    
    y_pred_train, y_train = reco_model.predict1(R, train_irow, train_jcol)
    
    
    
    st.subheader("Prediction for Train Set:") 
    
    df = pd.DataFrame(dict(train_irow= train_irow ,train_jcol = train_jcol, y_pred_train =y_pred_train, y_train=y_train))
    st.dataframe(df)
    
    st.subheader("Performance Metrics for Train Set:") 
    
    st.write(f"Mean Squared Error: {mean_squared_error(y_train,y_pred_train)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y_train,y_pred_train)}")
    st.write(f"R^2 score: {r2_score(y_train,y_pred_train)}")
    
    
    
    st.subheader("Prediction for Test Set") 
    
    df = pd.DataFrame(dict(test_irow= test_irow ,test_jcol = test_jcol, y_pred=y_pred, y=y))
    st.dataframe(df)
    
    st.subheader("Performance Metrics for Test Set:") 
    
    st.write(f"Mean Squared Error: {mean_squared_error(y,y_pred)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y,y_pred)}")
    st.write(f"R^2 score: {r2_score(y,y_pred)}")
    
    
    st.header("Model 2(with Regularization)") 
    
    
    #separating train and test set 
    irow, jcol = np.where(~np.isnan(R))
    idx = np.random.choice(np.arange(100000), 20000, replace=False)
    test_irow = irow[idx]
    test_jcol = jcol[idx]
    
    R_copy = R.copy()
    
    for i, j in zip(test_irow, test_jcol):
            R_copy[i][j] = np.nan
            
    r = R_copy.copy()
    st.subheader("Train set")
    st.dataframe(r)
    
    #model fitting 
    reco_model = reco_sys_with_reg.recoSysWithReg( alpha = 0.02, iteration  = 500, lam = 1)
    reco_model.fit(r)
    y_pred, y= reco_model.predict1(R, test_irow, test_jcol)
    
    
    
    
    st.subheader("Prediction for Train Set:") 
    
    train_irow, train_jcol = np.where(~np.isnan(R_copy))
    y_pred_train, y_train = reco_model.predict1(R, train_irow, train_jcol)
    df = pd.DataFrame(dict(train_irow= train_irow ,train_jcol = train_jcol, y_pred_train =y_pred_train, y_train=y_train))
    st.dataframe(df)
    
    st.subheader("Performance Metrics for Train Set:") 
    
    st.write(f"Mean Squared Error: {mean_squared_error(y_train,y_pred_train)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y_train,y_pred_train)}")
    st.write(f"R^2 score: {r2_score(y_train,y_pred_train)}")
    
    
    st.subheader("Prediction for Test Set:") 
    
    df = pd.DataFrame(dict(test_irow= test_irow ,test_jcol = test_jcol,y_pred=y_pred, y=y))
    st.dataframe(df)
    
    st.subheader("Performance Metrics for Test Set:") 
    
    st.write(f"Mean Squared Error: {mean_squared_error(y,y_pred)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y,y_pred)}")
    st.write(f"R^2 score: {r2_score(y,y_pred)}")
    
    
    st.header("Finding best paramete for lamda with validation set  ") 
    st.latex(r"\lambda = {0.8}")
    
    
    st.write("Since parameter optimization part takes too much time, I have directly wrote the best lambda")
    
    st.markdown("""
        Before the starting the finding best parameter process, I have separated the data to test and 
        train set. I stored test set separately and then for each lambda configuration I separated the 
        train set into ,train and validation sets. Then by using performance on the validation set I have
        found out that best lambda parameter is 0.8 . Here is the perfomance of the best lambda in test set.
                """)
    
    
    #separating train , validation and test set 
    irow, jcol = np.where(~np.isnan(R))
    idx = np.random.choice(np.arange(100000), 20000, replace=False)
    test_irow = irow[idx]
    test_jcol = jcol[idx]
    
    R_copy = R.copy()
    
    for i, j in zip(test_irow, test_jcol):
            R_copy[i][j] = np.nan
           
    r =  R_copy.copy() 
    # =============================================================================
    # 
    # best_lamda = 0
    # best_mse = 9999999
    # for i in range(1,20,1):
    #     lamda = i/10
    #     #separating train and test set 
    #     irow, jcol = np.where(~np.isnan(R_copy))
    #     idx = np.random.choice(np.arange(80000), 20000, replace=False)
    #     val_irow = irow[idx]
    #     val_jcol = jcol[idx]
    # 
    #     R_train = R_copy.copy()
    # 
    #     for i, j in zip(val_irow, val_jcol):
    #             R_train[i][j] = np.nan
    #             
    #     r = R_train.copy()
    #     st.subheader("Train set")
    #     st.dataframe(r)
    # 
    #     #model fitting 
    #     reco_model = reco_sys_with_reg.recoSysWithReg( alpha = 0.02, iteration  = 500, lam = lamda)
    #     reco_model.fit(r)
    #     y_pred, y= reco_model.predict1(R, val_irow, val_jcol)
    # 
    #     mse = mean_squared_error(y,y_pred)
    #     if mse < best_mse :
    #         best_mse = mse
    #         best_lamda = lamda
    # 
    # print(f"best lambda : {best_lamda} , best mse in validation set :{best_mse}  ")
    # 
    # st.subheader("performance of lambda in test set ") 
    # =============================================================================
    
    
    
    st.subheader("Prediction for Test Set with Best Lambda:") 
    
    #model fitting 
    reco_model = reco_sys_with_reg.recoSysWithReg( alpha = 0.02, iteration  = 500, lam = 0.8)
    reco_model.fit(r)
    y_pred, y= reco_model.predict1(R, test_irow, test_jcol)
    
    df = pd.DataFrame(dict(test_irow= test_irow ,test_jcol = test_jcol, y_pred=y_pred, y=y))
    st.dataframe(df)
    
    st.subheader("Performance Metrics for Test Set with Best Lambda:") 
    
    st.write(f"Mean Squared Error: {mean_squared_error(y,y_pred)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y,y_pred)}")
    st.write(f"R^2 score: {r2_score(y,y_pred)}")
    
