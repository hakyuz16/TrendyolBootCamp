#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:38:29 2022

@author: humeyraakyuz
"""

import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def root_mean_squared_error(actual, predictions):
    return np.sqrt(mean_squared_error(actual, predictions))



class recoSysWithReg:
    b_user: np.ndarray
    b_item: np.ndarray
    objective_values: np.ndarray
    err_values: np.ndarray
    def __init__(self , alpha: any = 0.01, iteration : any = 2000, lam: any = 0.5):
        """
        :param lam:
        :param alpha: 
        :param iteration:
        :param sigma:
        """
        self.lam = lam
        self.alpha = alpha
        self.iteration = iteration
      

    def fit(self, r: np.ndarray):
        m, n = r.shape
        self.objective_values=[]
        self.err_values=[]
        self.b_user= np.random.random((m))
        self.b_item= np.random.random((n))
        irow,icol=np.where(~np.isnan(r))
        
        for i in trange(self.iteration):
            g_b_user = np.zeros(m)
            g_b_item = np.zeros(n)
            
            objective = 0
            err = 0
            for user in range(m):
                g_b_i = 0
                idx = np.where(~np.isnan(r[user]))
                y_i = r[user,idx]
                y_pred_b_i: np.ndarray = self.b_user[user] + self.b_item[idx]
                objective += np.sum((y_i - y_pred_b_i)**2) + self.lam*(self.b_user[user]**2)/2
                err_i = np.sum(y_i - y_pred_b_i)
                
                g_b_i = g_b_i + 2 * err_i  
                g_b_i = (g_b_i / len(idx)) + (self.lam * self.b_user[user])
                g_b_user[user] = g_b_user[user] - self.alpha * g_b_i
                
                err += err_i
                
            for item in range(n):
                g_b_j = 0
                idx = np.where(~np.isnan(r[:,item]))
                y_j = r[idx,item]
                y_pred_b_j: np.ndarray = self.b_user[idx] + self.b_item[item]
                objective += np.sum((y_j - y_pred_b_j)**2 ) + self.lam*(self.b_item[item]**2)/2
                err_j = np.sum(y_j - y_pred_b_j)
                g_b_j = g_b_j + 2 * err_j  
                g_b_j = (g_b_j / len(idx)) + (self.lam* self.b_item[item])
                g_b_item[item] = g_b_item[item] - self.alpha * g_b_j
                err += err_j
            
            objective = (objective / len(irow)) 
            self.objective_values.append(objective)
            self.err_values.append(err)
            
            self.b_user = self.b_user - self.alpha * g_b_user 
            self.b_item = self.b_item - self.alpha * g_b_item 
            #print(f"obj: {objective} err: {err}")
            
        # plot objective function during iterations
        plt.figure(figsize = (10, 6))
        plt.plot(range(1, self.iteration + 1), self.err_values, "k-")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()
        
        return self.b_user,self.b_item

    def predict(self, r: np.array, user: int, item: int) -> np.ndarray:
        """
        :param r: Rating matrix
        :param user: User u
        :param item: item j
        :return: Calculated Rating of user rating for item j
        """

        score =  self.b_user[user] + self.b_item[item]

        return score

    def predict1(self, r: np.array, test_irow: np.ndarray,test_jcol: np.ndarray ) -> float:
        
        err = []
        y_pred_list= []
        y_list = []
        for u, j in zip(test_irow, test_jcol):
            y_pred = self.b_user[u] + self.b_item[j]
            y = r[u, j]
            y_pred_list.append(y_pred)
            y_list.append(y)
            err.append((y_pred - y) ** 2)
        print(np.sum(err)/len(test_irow))
       

        return y_pred_list,y_list