#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:43:29 2022

@author: humeyraakyuz
"""

import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import  scipy.signal.signaltools
from sklearn import preprocessing

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



def reg(x, y, group, p=0.3, verbose=False):
    beta = np.random.random(2)
    gamma = dict((k, np.random.random(2)) for k in range(6))

    if verbose:
        st.write(beta)
        st.write(gamma)
        st.write(x)

    alpha = 0.002
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _k, _x, _y in zip(group, x, y):
            y_pred = p * (beta[0] + beta[1] * _x) + (1 - p) * (gamma[_k][0] + gamma[_k][1] * _x)

            g_b0 = -2 * p * (_y - y_pred)
            g_b1 = -2 * p * ((_y - y_pred) * _x)

            # st.write(f"Gradient of beta0: {g_b0}")

            g_g0 = -2 * (1 - p) * (_y - y_pred)
            g_g1 = -2 * (1 - p) * ((_y - y_pred) * _x)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            gamma[_k][0] = gamma[_k][0] - alpha * g_g0
            gamma[_k][1] = gamma[_k][1] - alpha * g_g1

            err += (_y - y_pred) ** 2

        print(f"{it} - Beta: {beta}, Gamma: {gamma}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta, gamma





def modifiedModel(x, y,teta) -> np.ndarray:
    
    lam=0.01
    alpha=0.001
    iteration=200
    
    
    objective_values=[]
    beta = np.random.random(2)
 
    for i in range(iteration):
        
        y_pred: np.ndarray = beta[0] + beta[1] * x
        
        g_b0 = 0
        g_b1 = 0
        objective = 0
        for i,v in enumerate(x):
            objective += (max(teta,np.abs(y[i] - y_pred[i])))**2 
            if  np.abs(y[i] - y_pred[i]) >= teta:
                
                
                err = y[i] - y_pred[i]
                g_b0 = g_b0 -2 * err
                g_b1 = g_b1 -2 * err * v
            
        
        g_b0 = (g_b0 / len(x))  +  2 * lam * beta[0]
        g_b1 = (g_b1 / len(x))  + 2 * lam * beta[1]
        
        objective = (objective / len(x) + ((beta[0]+ beta[1])**2))
        
        
        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1} obj: {objective}")
        beta_prev = np.copy(beta)
        objective_values.append(objective)

        
        beta[0] = beta[0] - alpha * g_b0 
        beta[1] = beta[1] - alpha * g_b1 

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break
        
    # plot objective function during iterations
    plt.figure(figsize = (10, 6))
    plt.plot(range(1, iteration + 1), objective_values, "k-")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()
 
    return beta




def normalize(df):
       for cl in df:
           df[cl]= (df[cl]- df[cl].mean())/df[cl].std()
       return df


def performance(beta,X,y):
    y_pred=[]
    for v in X:
         y_pred.append( beta[0] + beta[1] * v)

    dg1 = pd.DataFrame(dict(x=X, y=y, y_pred=y_pred))


    fig1 = plt.figure(figsize = (10, 10))
    plt.plot(X, dg1["y"], "b.", markersize = 5)
    plt.plot(X, dg1["y_pred"], "c.", markersize = 5)
    st.plotly_chart( fig1, use_container_width=True)


    st.write(f"Mean Squared Error: {mean_squared_error(y,y_pred)}")
    st.write(f"Root Mean Squared Error: {root_mean_squared_error(y,y_pred)}")
    st.write(f"R^2 score: {r2_score(y,y_pred)}")




st.header("Building a Simple Regression Model with Different Loss Function")
st.markdown("""
 In the problem we assume below the some threshold error does not effect the performance:
 
 For example for a house with 5 price if prediction is between 4 and 6, prediction is okay,
 otherwise we should penalize deviation.
 In order to achieve this our objective function(loss function is changed)
 New loss function is: 
 """)

st.subheader("Formulating the Model")
st.markdown("#### General Model")
st.latex(r"\hat{y}^{0}_i=\beta_0 + \beta_1 x_i")

st.markdown("#### Loss Function")


st.latex(
    r"L(\beta_0,\beta_1)=\sum_{i=1}^{N}{(max(\Theta,(y_i - \hat{y}_i ))^2)/N + \lambda (\beta_0^2 + \beta_1^2)}")


st.markdown("#### Partial derivatives")

st.markdown("""
    we  should use chain rule  and backpropagation to explain the max function effect
    
    for example derivative of that function with respect to lambda is:
            """)
st.latex(
    r" y(\lambda) = max(\Theta,\lambda)")
            
st.latex(
    r"\frac{\partial y(\lambda)}{\partial \lambda} = \begin{cases} 1 &\text{if } \lambda > \Theta \\ 0 &\text{otherwise } \end{cases}")
            

st.markdown("""Therefore if error is smaller than a specified threshold , it willl not effect the gradient 
            for our cases we can calculate gradient like that:""")

st.latex( r"\qquad \qquad \Delta B_0 = 0  , \Delta B_1 = 0")
st.latex( r"\text{for i in X:} ")
st.latex( r"\qquad \qquad \qquad \qquad \qquad \text{if } y_i -  \hat{y}_i > \Theta :")
st.latex( r"\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Delta B_0 + = -2 * (y_i -  \hat{y}_i) ")
st.latex( r"\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Delta B_1 + = -2 * (y_i -  \hat{y}_i) * x_i ")
st.latex( r"\qquad \qquad \qquad  \Delta B_1 =  \frac {\Delta B_1}{N} + 2 *  \lambda * \beta_1 ")
st.latex( r"\qquad \qquad \qquad  \Delta B_0 =  \frac {\Delta B_0}{N} + 2 *  \lambda * \beta_0 ")



st.header("Dataset")

# fetch the data
cal_housing = fetch_california_housing()
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
HouseAgeGroup= (df['HouseAge'].values / 10).astype(np.int)

#clear not related features
df = pd.DataFrame(dict(MedInc=df['MedInc'], price=cal_housing.target))

# if it is required, normalize the data
# for that case , normalization does not work well
#df=normalize(df) 


x,y= np.array(df["MedInc"]),np.array(df["price"])
st.dataframe(df)

st.subheader("Correlation")
## correlation between income and house price is 0.68
r = np.corrcoef(x, y)

st.write(f"Correlation between Income and House Price is : { r[0][1] } ")


#split the data set to train and test
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)


p = st.slider("Mixture Ration (p)", 0.0, 1.0, value=0.8)
beta, gamma = reg(X_train,y_train,HouseAgeGroup,
                  p=p,
                  verbose=False)

st.subheader(f"General Model with p={p:.2f} contribution")
st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")


st.header("Performance metrics for general model ")

st.header("Train performance")
performance(beta, X_train, y_train)

st.header("Test Performance")
performance(beta, X_test, y_test)

st.header("Performance Metrics for Modified Model ")

teta = st.slider("teta value", 0.0, 1.0, value=0.4)

st.header("Train performance")
#apply simple linear regression (L2 regularize)
w = modifiedModel(X_train,y_train,teta)
performance(w, X_train, y_train)

st.header("Test performance")
performance(w, X_test, y_test)

