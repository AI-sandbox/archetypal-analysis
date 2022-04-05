# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:39:27 2019

@author: Benyamin Motevalli

This class is developed based on "Archetypa Analysis" by Adele Cutler and Leo
Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347

"""

import numpy as np
from scipy.optimize import nnls      
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.matlib import repmat
from math import *
import time

class ArchetypalAnalysis():
    
    """
    Parameters:
    -----------
           
    n_archetypes:   Defines the number of archetypes.
    n_dim:          Number of features (Dimensions).
    n_data:         Number of data in dataset.
    
    X:          
        Dimension:  n_dim x n_data
        
                    Array of data points. It is the transpose of input data.
                    
    
    archetypes:
        
        Dimension:  n_dim x n_archetypes
        
                    Array of archetypes. Each columns represents an archetype.
    
        
    alfa:
        Dimension:  n_archetypes x n_data
        
                    Each column defines the weight coefficients for each 
                    archetype to approximate data point i.
                    Xi = sum([alfa]ik x Zk)
    
    beta:
        Dimension:  n_data x n_archetypes
        
                    Each column defines the weight coefficients for each data 
                    point from which archetype k is constructed.
    
    tolerance:      Defines when to stop optimization.

    max_iter:       Defines the maximum number of iterations

    random_state:   Defines the random seed number for initialization. No effect if "furthest_sum" is selected.       
                    
    C:              is a constraint coefficient to ensure that the summation of
                    alfa's and beta's equals to 1. C is conisdered to be inverse
                    of M^2 in the original paper.

    initialize:     Defines the initialization method to guess initial archetypes:

                        1. furthest_sum (Default): the idea and code taken from https://github.com/ulfaslak/py_pcha and the original author is: Ulf Aslak Jensen.
                        2. random:  Randomly selects archetypes in the feature space. The points could be any point in space
                        3. random_idx:  Randomly selects archetypes from points in the dataset.
    
    """
    
    def __init__(self, n_archetypes = 2, tolerance = 0.001, max_iter = 200, random_state = 0, C = 0.0001, initialize = 'furthest_sum', redundancy_try = 30):
        
       
        self.n_archetypes = n_archetypes
        self.n_dim = None
        self.n_data = None
        self.tolerance = tolerance
        self.C = C
        self.random_state = random_state
        self.archetypes = []
        self.alfa = []
        self.beta = []
        self.explained_variance_ = []
        self.RSS_ = None
        self.RSS0_ = None
        self.RSSi_ = []
        self.initialize = initialize.lower()
        self.close_match = {}
        self.max_iter = max_iter
        self.redundancy_try = redundancy_try
    
    def fit(self, X):
        
        self.X = X.T
        self.n_dim, self.n_data = self.X.shape
        self.RSS0_ = (self.X.var(axis = 1)).sum()
        self.RSS0_2 = (self.X.var(axis = 1) * self.n_data).sum()
        
        if (self.n_archetypes == 1):
            self.archetypes = self.X.mean(axis = 1).reshape(self.n_dim, self.n_archetypes)
            self.alfa = np.ones([self.n_archetypes, self.n_data])
            self.RSS_ = self.RSS0_
            self.RSS_2 = self.RSS0_2
            
        else:
            
            self._initialize_archetypes()
            
            self.RSS_ = 1.0
            RSS_old = 100.0
            
            self.count_converg_ = 1
            
            while ((abs(self.RSS_ - RSS_old)) / RSS_old > self.tolerance):
                
#                print('old = ', RSS_old, ' new = ', self.RSS_, 'err = ', abs(self.RSS_ - RSS_old))
                RSS_old = self.RSS_
                
                self._optimize_alfa()
                has_error = self._optimize_beta()

                if(has_error):
                    return
                
                self.archetypes = np.matmul( self.X, self.beta)
                
                self.RSS_ = self.RSSi_.sum()   

                if (self.count_converg_ > self.max_iter):
                    break
                
                self.count_converg_ += 1
                        
            self._rank_archetypes()
                
        self.X_approx = np.matmul(self.archetypes , self.alfa)   
        self.RSS_2 = calc_SSE(self.X, self.X_approx)
        self.explained_variance_ = explained_variance(self.X, self.X_approx, method = "_")
        self._extract_archetype_profiles()
        
    
    def fit_transform(self,X):
        
        self.fit(X)
        
        X_approx = np.matmul(self.archetypes , self.alfa)
        
        
        return X_approx.T
    
    def transform(self, X_new):
        
        X_new_Trans = X_new.T
        
        n_data, n_dim = X_new.shape
        
        alfa_new, RSSi_new = self.__optimize_alfa_for_transform(X_new, n_data)
#        self.alfa, self.RSSi_ = self.__optimize_alfa_for_transform(X_new, n_data)
        
        X_new_Approx = np.matmul(self.archetypes , alfa_new)
        
        return X_new_Approx.T, alfa_new
    
    
    
    def _initialize_archetypes(self):
        
        if self.initialize == 'random':
            self._random_initialize()
            
        elif self.initialize == 'random_idx':
            self._random_idx_initialize()
            
        elif self.initialize == 'furthest_sum':
            try:
                self._furthest_sum_initialize()
            except IndexError:
                class InitializationException(Exception): pass
                raise InitializationException("Initialization with furthest sum does not converge. Too few examples in dataset.\n"
                                              + "A random initialization is selected to continue.")
                self._random_initialize()
            
            
    def _random_initialize(self):
                   
        from sklearn.preprocessing import MinMaxScaler
        
        np.random.seed(self.random_state)
        
        sc = MinMaxScaler()
        sc.fit(self.X.T)
        
        archs_norm = np.random.rand(self.n_archetypes, self.n_dim)
        
        self.archetypes_init_ = sc.inverse_transform(archs_norm).T
        self.archetypes = copy(self.archetypes_init_)
        
        
    def _furthest_sum_initialize(self):
                   
        init = [int(np.ceil(len(range(self.n_data)) * np.random.rand()))]
        n_data_range = range(self.n_data)
        init_arch_idx = furthest_sum(self.X[:, n_data_range], self.n_archetypes, init)
        
        self.archetypes_init_ = self.X[:,init_arch_idx]
        self.archetypes = copy(self.archetypes_init_)
        
    
    def _random_idx_initialize(self):
        
        lst_idx = []
        np.random.seed(self.random_state)
        #np.random.seed(5)
        for i in range(self.n_archetypes):
            lst_idx.append(np.random.randint(self.n_data))
            
        
        self.archetypes_init_ = self.X[:,lst_idx]
        self.archetypes = copy(self.archetypes_init_)


    
    def __optimize_alfa_for_transform(self, X_new, n_data):
        
        """
        This functions aims to obtain corresponding alfa values for a new data
        point after the fitting is done and archetypes are determined.
        
        Having alfas, we can approximate the new data-points in terms of 
        archetypes.
        
        NOTE: X_new dimension is n_data x n_dim. Here, the original data is passed
        instead of transpose.
        """
        
        alfa = np.zeros([self.n_archetypes, n_data])
        RSSi_ = np.zeros([n_data])
        for i, xi in enumerate(X_new):
            alfa[:, i], RSSi_[i] = solve_constrained_NNLS(xi, self.archetypes, self.C)
            
        return alfa, RSSi_
    
    
    def _optimize_alfa(self):
        
        """
        self.archetypes: has a shape of n_dim x n_archetypes
        self.alfai:      has a shape of n_archetypes x 1.
        xi:              has a shape of n_dim x 1.
        
        The problem to minimize is:
            
            xi = self.archetypes x self.alfai
        """
        
        self.alfa = np.zeros([self.n_archetypes, self.n_data])
        self.RSSi_ = np.zeros([self.n_data])
        for i, xi in enumerate(self.X.T):            
            self.alfa[:, i], self.RSSi_[i] = solve_constrained_NNLS(xi, self.archetypes, self.C)


    def _optimize_beta(self):
        
        self.beta = np.zeros([self.n_data, self.n_archetypes])
        for l in range(self.n_archetypes):
            
            v_bar_l, has_error = self._return_vbar_l(l)

            if has_error:
                return has_error

            v_bar_l = v_bar_l.flatten()
            self.beta[:,l], _ = solve_constrained_NNLS(v_bar_l, self.X,self.C)
        
        return has_error

    
    def _find_new_archetype(self, k):
    
        """
        In some circumstance, summation of alfa's for an archetype k becomes zero. 
        That means archetype k is redundant. This function aims to find a new candidate
        from data set to replace archetype k.        
        """

        arch_k = copy(self.archetypes[:,k])

        if k == 0:
            alfa_k = self.alfa[k:,:]
            archetypes = self.archetypes[:,k:]

        else:
            alfa_k = self.alfa[k-1:,:]
            archetypes = self.archetypes[:,k-1:]

        X_approx = np.matmul(archetypes , alfa_k)

        RSS_i = ((self.X.T - X_approx.T) ** 2).sum(axis=1)
        if self.n_data < 10:
            id_maxes = RSS_i.argsort()
        else:
            id_maxes = RSS_i.argsort()[-10:]

        if np.linalg.norm(self.X.T[id_maxes[-1],:] - arch_k):
            id_max = id_maxes[-1]
        else:
            import random
            id_max = random.Random(self.random_state).choice(list(id_maxes[:-1]))
            self.random_state += 10

        return self.X.T[id_max,:], id_max
  

    def _return_vbar_l(self, l):        
        
        def return_vi(i, l, alfa_il):
            """
            This function calculates vi for each data point with respect to 
            archetype l.
            
            i:          ith data point (xi)
            l:          index of archetype that should be excluded.
            """
            eps = 0.000001
            
            
            vi = np.zeros([self.n_dim,1])
            # for k, alfaik, zk in enumerate(zip(self.alfa[:,i], self.archetypes)):
            for k in range(self.n_archetypes):                
                if k != l:
                    xx = self.alfa[k,i] * self.archetypes[:,k]
                    vi[:,0] = vi[:,0] + xx
            
            if (alfa_il < eps):
                alfa_il = eps
            vi[:,0] = (self.X[:,i] - vi[:,0]) / alfa_il
            
            return vi
        
        
        def check_arch_redundancy():
            
            has_error = False
            
            eps = 0.00000001
            
            # CHECK SUM SQ OF alfa_il
            
            alfa_il_sumsq = np.sum(self.alfa[l,:] ** 2)
            
            count = 1
            while(alfa_il_sumsq < eps):
                
                # FINDING THE FURTHEST POINT AND REPLACING REDUNDANT ARCHETYPE l
                # arch_old = copy(self.archetypes[:,l])
                self.archetypes[:,l], id_max  = self._find_new_archetype(l)
                # print(np.linalg.norm(arch_old - self.archetypes[:,l]))
                
                # RE-OPTIMIZING ALFA
                self._optimize_alfa()
                # self.alfa[l,:], _ = self.__optimize_alfa_for_transform(self.archetypes[:,l].reshape(1,-1), n_data = 1)
                
                # RE-CALCULATE SUM SQ OF alfa_il
                alfa_il_sumsq = np.sum(self.alfa[l,:] ** 2)
                
                if (count > self.redundancy_try):
                    has_error = True
                    break
                
                count += 1
            if (count > 1):
                # print(f'Warning: Archetype {l+1} was recognised redundant. The redundancy issue was resolved after {count -1} try(s) and a new candidate is replaced.')
                print(f'Warning: A redundant archetype was recognised. The redundancy issue was resolved after {count -1} try(s) and a new candidate is replaced.')
                
            return has_error, alfa_il_sumsq
        
        
        has_error, alfa_il_sumsq = check_arch_redundancy()
        
        if (has_error):
            self.n_archetypes = self.n_archetypes - 1
            print(f'Warning: After {self.redundancy_try} tries, the redundancy issue was not resolved. Hence, the number of archetypes is reduced to: {self.n_archetypes}')                    
            self.fit(self.X.T)  
            return None, has_error
        else:      
            eps = 0.000001
            sum_alfa_il_sq_vi = np.zeros([self.n_dim,1])
            for i in range(self.n_data):
                alfa_il = self.alfa[l,i]                
                vi = return_vi(i,l,alfa_il)
                sum_alfa_il_sq_vi = sum_alfa_il_sq_vi + alfa_il ** 2 * vi
            
            if (alfa_il_sumsq < eps):
                alfa_il_sumsq = eps
            v_bar = sum_alfa_il_sq_vi / alfa_il_sumsq
        
        return v_bar, has_error
    
    
    def _rank_archetypes(self):
        
        """
        This function aims to rank archetypes. To do this, each data point is 
        approximated just using one of the archetypes. Then, we check how good
        is the approximation by calculating the explained variance. Then, we 
        sort the archetypes based on the explained variance scores. Note that,
        unlike to PCA, the summation of each individual explained variance 
        scores will not be equal to the calculated explained variance when all
        archetypes are considered.
        """
        exp_var_per_arch_ = []
        for i in range(self.n_archetypes):
            X_approx = np.matmul(self.archetypes[:,i].reshape(self.n_dim,1) , self.alfa[i,:].reshape(1,self.n_data))
            exp_var_per_arch_.append((i,explained_variance(self.X, X_approx, method = "_")))
    
#        self.exp_var_per_arch_ = tuple(self.exp_var_per_arch_)
        
        exp_var_per_arch_ = sorted(exp_var_per_arch_, key = lambda x: x[1], reverse=True )
        rank = [item[0] for item in exp_var_per_arch_]
        self.score_per_arch = [item[1] for item in exp_var_per_arch_]
        
        self.archetypes = self.archetypes[:,rank]
        self.alfa = self.alfa[rank,:]
        self.beta = self.beta[:,rank]
   
    
    def plot_simplex(self, alfa, plot_args = {}, grid_on = True):
        
        """
        # group_color = None, color = None, marker = None, size = None
        group_color:    
            
            Dimension:      n_data x 1
            
            Description:    Contains the category of data point.
        """
        
        labels = ('A'+str(i + 1) for i in range(self.n_archetypes))
        rotate_labels=True
        label_offset=0.10
        data = alfa.T
        scaling = False
        sides=self.n_archetypes
        
        basis = np.array(
                    [
                        [
                            np.cos(2*_*pi/sides + 90*pi/180),
                            np.sin(2*_*pi/sides + 90*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )
    
        # If data is Nxsides, newdata is Nx2.
        if scaling:
            # Scales data for you.
            newdata = np.dot((data.T / data.sum(-1)).T,basis)
        else:
            # Assumes data already sums to 1.
            newdata = np.dot(data,basis)
    
        
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
    
        for i,l in enumerate(labels):
            if i >= sides:
                break
            x = basis[i,0]
            y = basis[i,1]
            if rotate_labels:
                angle = 180*np.arctan(y/x)/pi + 90
                if angle > 90 and angle <= 270:
                    angle = (angle + 180) % 360 # mod(angle + 180,360)
            else:
                angle = 0
            ax.text(
                    x*(1 + label_offset),
                    y*(1 + label_offset),
                    l,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=angle
                )
    
        # Clear normal matplotlib axes graphics.
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_frame_on(False)
        
        
        # Plot border
        lst_ax_0 = []
        lst_ax_1 = []
        ignore = False
        for i in range(sides):
            for j in range(i + 2, sides):
                if (i == 0 & j == sides):
                    ignore = True
                else:
                    ignore = False                        
    #                
                if not (ignore):                    
                    lst_ax_0.append(basis[i,0] + [0,])
                    lst_ax_1.append(basis[i,1] + [0,])
                    lst_ax_0.append(basis[j,0] + [0,])
                    lst_ax_1.append(basis[j,1] + [0,])
    

        
        ax.plot(lst_ax_0,lst_ax_1, color='#FFFFFF',linewidth=1, alpha = 0.5, zorder=1)
        
        # Plot border
        lst_ax_0 = []
        lst_ax_1 = []
        for _ in range(sides):
            lst_ax_0.append(basis[_,0] + [0,])
            lst_ax_1.append(basis[_,1] + [0,])
    
        lst_ax_0.append(basis[0,0] + [0,])
        lst_ax_1.append(basis[0,1] + [0,])

    #    
        ax.plot(lst_ax_0,lst_ax_1,linewidth=1, zorder=2) #, **edge_args ) 
        
        if len(plot_args) == 0:
            ax.scatter(newdata[:,0], newdata[:,1], color='black', zorder=3, alpha=0.5)
        else:
            if ('marker' in plot_args):   
                marker_vals = plot_args['marker'].values
                marker_unq = np.unique(marker_vals)                
                
                for marker in marker_unq:
                    row_idx = np.where(marker_vals == marker)
                    tmp_arg = {}
                    for keys in plot_args:
                        if (keys!= 'marker'):
                            tmp_arg[keys] = plot_args[keys].values[row_idx]
                    
                    ax.scatter(newdata[row_idx,0],newdata[row_idx,1], **tmp_arg, marker =  marker, alpha=0.5, zorder=3)
            else:
                ax.scatter(newdata[:,0], newdata[:,1], **plot_args, marker = 's', zorder=3, alpha=0.5)
        
        
        
        plt.show()
            
               
    def parallel_plot(self, lst_feat, df_color, linewidth = '0.3', arch_color = 'black'):
        
        """
        Based on source: http://benalexkeen.com/parallel-coordinates-in-matplotlib/
        
        lst_feat:
                    list of features.
        
        df_color:
                    A dataframe of collection of colors corresponding to each
                    data point.
        
        """
    
        from matplotlib import ticker
        
        x = [i for i, _ in enumerate(lst_feat)]
        
        df = pd.DataFrame(self.X.T, columns = lst_feat)
        
        for i in range(self.n_archetypes):
            df.loc[-1] = list(self.archetypes[:,i])
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index            
            
            df_color.loc[-1] = 'arch'
            df_color.index = df_color.index + 1  # shifting index
            df_color = df_color.sort_index()  # sorting by index          
                
        # Create (X-1) sublots along x axis
        fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))
        
        # Get min, max and range for each column
        # Normalize the data for each column
        min_max_range = {}       
        for col in lst_feat:
            min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
            df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
        
        # Plot 
        for i, ax in enumerate(axes):
            for idx in df.index:    
                if (df_color.loc[idx,'color'] == 'arch'):
                    # Plot each archetype
                    ax.plot(x, df.loc[idx, lst_feat], color = arch_color, alpha = 0.8, linewidth='2.0')
                else:
                    # Plot each data point
                    ax.plot(x, df.loc[idx, lst_feat], color = df_color.loc[idx,'color'], alpha = 0.3, linewidth=linewidth)
            ax.set_xlim([x[i], x[i+1]])            
            
        # Set the tick positions and labels on y axis for each plot
        # Tick positions based on normalised data
        # Tick labels are based on original data
        def set_ticks_for_axis(dim, ax, ticks):
            min_val, max_val, val_range = min_max_range[lst_feat[dim]]
            step = val_range / float(ticks-1)
            tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
            norm_min = df[lst_feat[dim]].min()
            norm_range = np.ptp(df[lst_feat[dim]])
            norm_step = norm_range / float(ticks-1)
            ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
            ax.yaxis.set_ticks(ticks)
            ax.set_yticklabels(tick_labels)
        
        for dim, ax in enumerate(axes):
            ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
            set_ticks_for_axis(dim, ax, ticks=6)
            ax.set_xticklabels([lst_feat[dim]], rotation='vertical')
            
        
        # Move the final axis' ticks to the right-hand side
        ax = plt.twinx(axes[-1])
        dim = len(axes)
        ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([lst_feat[-2], lst_feat[-1]], rotation='vertical')
        
        
        # Remove space between subplots
        plt.subplots_adjust(wspace=0)       
        plt.show()
        
    
    def _extract_archetype_profiles(self):
        """
        This function extracts the profile of each archetype. Each value in
        each dimension of archetype_profile shows the portion of data that
        is covered by that archetype in that specific direction.
        """
        self.archetype_profile = np.zeros([self.n_dim, self.n_archetypes])
        for i in range(self.n_archetypes):
            for j in range(self.n_dim):
                
                x_arch = self.archetypes[j,i]
                x_data = self.X[j, :]
                
                self.archetype_profile[j,i] = ecdf(x_data, x_arch)
                
    
    def plot_profile(self, feature_cols = None):
        """
        This function plots the profile of the archetypes.
        
        feature_cols:
            Optional input. list of feature names to use to label x-axis.
        """               
        
        plt.style.use('ggplot')
        
        x_vals = np.arange(1, self.n_dim + 1)
        
        
        for i in range(self.n_archetypes):
            plt.figure(figsize=(14,5))            
            plt.bar(x_vals, self.archetype_profile[:,i] * 100.0, color = '#413F3F')
            if (feature_cols != None):
                plt.xticks(x_vals, feature_cols, rotation='vertical')
            plt.ylim([0,100])
            plt.ylabel('A' + str(i + 1))
            plt.rcParams.update({'font.size': 10})
            plt.tight_layout()
            
            
    
    def plot_radar_profile(self, feature_cols = None, Title = 'Radar plot of archetype profiles'):
        
        if (feature_cols != None):
            labels = feature_cols
        else:
            labels = [str(i+1) for i in range(self.n_dim)]
            
        angles = np.linspace(0, 2*np.pi, self.n_dim, endpoint = False)
        angles = np.concatenate((angles, [angles[0]])) 
        
        fig=plt.figure()
        legend = []
        for i in range(self.n_archetypes):
            
            x_close = self.archetype_profile[:,i]
            x_close = np.concatenate((x_close, [x_close[0]]))           
            
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, x_close, 'o-', linewidth=2)
            ax.fill(angles, x_close, alpha=0.25)
            ax.set_thetagrids(np.array(angles) * 180.0/np.pi, labels)
            ax.set_title(Title)
            ax.grid(True)
            legend.append('A'+str(i + 1))
        
        ax.legend(legend)
        
        
            
    
    def _extract_closes_match(self):
        
        from scipy.spatial.distance import cdist
        
        self.close_match = {}
        for i in range(self.n_archetypes):
            r = cdist(self.X.T, self.archetypes[:,i].reshape(1, self.n_dim))
            i_min = np.argmin(r)
            
            self.close_match[i+1] = (i_min, self.alfa[:,i_min])
            
    
    def plot_close_match(self, Title = 'Radar plot of closest matches'):
        
#        import seaborn as sns
        
        if (len(self.close_match) == 0):
            self._extract_closes_match()
        
        labels = ['A' + str(i+1) for i in range(self.n_archetypes)]
        angles = np.linspace(0, 2*np.pi, self.n_archetypes, endpoint = False)
        angles = np.concatenate((angles, [angles[0]])) 
        
        fig=plt.figure()
        for i in range(1, self.n_archetypes + 1):
            
            x_close = self.close_match[i][1]
            x_close = np.concatenate((x_close, [x_close[0]]))           
            
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, x_close, 'o-', linewidth=2)
            ax.fill(angles, x_close, alpha=0.25)
            ax.set_thetagrids(np.array(angles) * 180.0/np.pi, labels)
            ax.set_title(Title)
            ax.grid(True)



def find_furthest_point(x_search, x_ref):
    
    """
    This function finds a data point in x_search which has the furthest 
    distance from all data points in x_ref. In the case of archetypes,
    x_ref is the archetypes and x_search is the dataset.
    
    Note:
        In both x_search and x_ref, the columns of the arrays should be the
        dimensions and the rows should be the data points.
    """
    from scipy.spatial.distance import cdist
    
    D = cdist(x_ref, x_search)
    D_sum = D.sum(axis=0)
    idxmax = np.argmax(D_sum)
    
    return x_search[idxmax,:], idxmax

                        


def ecdf(X, x):
    
    """Emperical Cumulative Distribution Function
    
    X: 
        1-D array. Vector of data points per each feature (dimension), defining
        the distribution of data along that specific dimension.
        
    x:
        Value. It is the value of the corresponding dimension of an archetype.
        
    P(X <= x):
        The cumulative distribution of data points with respect to the archetype
        (the probablity or how much of data in a specific dimension is covered
        by the archetype).
    
    """
    
    return float(len(X[X < x]) / len(X))


def calc_SSE(X_act, X_appr):
    
    """
    This function returns the Sum of Square Errors.
    """
    
    return ((X_act - X_appr) ** 2).sum()

def calc_SST(X_act):
    
    """
    This function returns the Sum of Square Errors.
    """
    
    return (X_act ** 2).sum()




def explained_variance(X_act, X_appr, method = 'sklearn'):
    
    
    if (method.lower == 'sklearn'):
            
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(X_act.T, X_appr.T)  
        
    else:
        
        SSE = calc_SSE(X_act, X_appr)
        SST = calc_SST(X_act)
        
        return (SST - SSE) / SST
        
        
def solve_constrained_NNLS(u, t, C):
    
    """
    This function solves the typical equation of ||U - TW||^2 where U and T are
    defined and W should be determined such that the above expression is 
    minimised. Further, solution of W is subjected to the following constraints:
        
        Constraint 1:       W >= 0
        Constraint 2:       sum(W) = 1
    
    Note that the above equation is a typical equation in solving alfa's and
    beta's.
    
    
    Solving for ALFA's:
    -------------------
    when solving for alfa's the following equation should be minimised:
        
        ||Xi - sum([alfa]ik x Zk)|| ^ 2.
    
    This equation should be minimised for each data point (i.e. n_data is the
    number of equations), which results in n_data rows of alfa's. In each 
    equation U, T, and W have the following dimensions:
        
    Equation (i):
        
        U (Xi):         It is a 1D-array of n_dim x 1 dimension.
        T (Z):          It is a 2D-array of n_dim x k dimension.
        W (alfa):       It is a 1D-array of k x 1 dimension.
    
    
    Solving for BETA's:
    -------------------   
    
    """
    
    m_observation, n_variables = t.shape
    
    # EMPHASIZING THE CONSTRAINT
    u = u * C
    t = t * C   
    
    # ADDING THE CONSTRAINT EQUATION
    u = np.append(u, [1], axis = 0)
    t = np.append(t, np.ones([1,n_variables]), axis = 0)
    w, rnorm = nnls(t, u)
    
    return w, rnorm
        
        

def ternaryPlot(data, scaling=True, start_angle=90, rotate_labels=True,
                labels=('one','two','three'), sides=3, label_offset=0.10,
                edge_args={'color':'black','linewidth':1},
                fig_args = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},
                grid_on = True
        ):
    '''
    source: https://stackoverflow.com/questions/701429/library-tool-for-drawing-ternary-triangle-plots
    
    This will create a basic "ternary" plot (or quaternary, etc.)
    
    DATA:           The dataset to plot. To show data-points in terms of archetypes
                    the alfa matrix should be provided.
    
    SCALING:        Scales the data for ternary plot such that the components along
                    each axis dimension sums to 1. This conditions is already imposed 
                    on alfas for archetypal analysis.
    
    start_angle:    Direction of first vertex.
    
    rotate_labels:  Orient labels perpendicular to vertices.
    
    labels:         Labels for vertices.
    
    sides:          Can accomodate more than 3 dimensions if desired.
    
    label_offset:   Offset for label from vertex (percent of distance from origin).
    
    edge_args:      Any matplotlib keyword args for plots.
    
    fig_args:       Any matplotlib keyword args for figures.
    
    '''
    basis = np.array(
                    [
                        [
                            np.cos(2*_*pi/sides + start_angle*pi/180),
                            np.sin(2*_*pi/sides + start_angle*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data,basis)

#    fig = plt.figure(**fig_args)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = angle = (angle + 180) % 360 # mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + label_offset),
                y*(1 + label_offset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False                        
#                
            if not (ignore):
#            if (j!=i & j!=i+1 & j != i-1):                        
                lst_ax_0.append(basis[i,0] + [0,])
                lst_ax_1.append(basis[i,1] + [0,])
                lst_ax_0.append(basis[j,0] + [0,])
                lst_ax_1.append(basis[j,1] + [0,])

#    lst_ax_0.append(basis[0,0] + [0,])
#    lst_ax_1.append(basis[0,1] + [0,])
    
    ax.plot(lst_ax_0,lst_ax_1, color='#FFFFFF',linewidth=1, alpha = 0.5)
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_,0] + [0,])
        lst_ax_1.append(basis[_,1] + [0,])

    lst_ax_0.append(basis[0,0] + [0,])
    lst_ax_1.append(basis[0,1] + [0,])
#    ax.plot(
#        [basis[_,0] for _ in range(sides) + [0,]],
#        [basis[_,1] for _ in range(sides) + [0,]],
#        **edge_args
#    )
#    
    ax.plot(lst_ax_0,lst_ax_1,linewidth=1) #, **edge_args ) 
    

    return newdata,ax 


def furthest_sum(K, noc, i, exclude=[]):
    """
    Note by Benyamin Motevalli:
        
        This function was taken from the following address:
            https://github.com/ulfaslak/py_pcha
            
        and the original author is: Ulf Aslak Jensen.
    """
    
    
    """
    Original Note:
        
    Furthest sum algorithm, to efficiently generat initial seed/archetypes.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).
    
    Parameters
    ----------
    K : numpy 2d-array
        Either a data matrix or a kernel matrix.

    noc : int
        Number of candidate archetypes to extract.

    i : int
        inital observation used for to generate the FurthestSum.

    exclude : numpy.1darray
        Entries in K that can not be used as candidates.

    Output
    ------
    i : int
        The extracted candidate archetypes
    """
    def max_ind_val(l):
        return max(zip(range(len(l)), l), key=lambda x: x[1])

    I, J = K.shape
    index = np.array(range(J))
    index[exclude] = 0
    index[i] = -1
    ind_t = i
    sum_dist = np.zeros((1, J), np.complex128)

    if J > noc * I:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, noc + 11):
            if k > noc - 1:
                Kq = np.dot(Kt[:, i[0]], Kt)
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = np.dot(Kt[:, ind_t].T, Kt)
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1
    else:
        if I != J or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = np.dot(Kt.T, Kt)
            K = np.lib.scimath.sqrt(
                repmat(np.diag(K), J, 1) - 2 * K + \
                repmat(np.mat(np.diag(K)).T, 1, J)
            )

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, noc + 11):
            if k > noc - 1:
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1

    return i

# =============================
# ADDITIONAL PLOTTING FUNCTIONS
# =============================

def compare_profile(AA_prof_1, AA_prof_2, feature_cols):
    """
    This function plots the profile of the archetypes.
    
    feature_cols:
        Optional input. list of feature names to use to label x-axis.
    """               
    
    plt.style.use('ggplot')
    
    n_dim = len(feature_cols)
    
    x_vals = np.arange(1, n_dim + 1)
    
    plt.figure(figsize=(14,5))
    
    plt.bar(x_vals, AA_prof_1 * 100.0, color = '#413F3F', label='Minimum Case')
    plt.bar(x_vals, AA_prof_2 * 100.0, color = '#8A2BE2', alpha = 0.5, label='Maximum Case')
    plt.xticks(x_vals, feature_cols, rotation='vertical')
    plt.ylim([0,100])
#    plt.ylabel('A' + str(i + 1))
    plt.rcParams.update({'font.size': 10})
    plt.tight_layout()
    plt.legend(loc='upper left')
    

                    
 
def datapoint_profile(x_point, x_data):
    
    point_profile = []
    
    for i, p in enumerate(x_point):
        
        d = x_data[i, :]
        
        point_profile.append(ecdf(d, p))
        
    return np.array(point_profile)


def plot_radar_datapoint(AA, X, Title = 'Radar plot of datapoint'):
        
#        import seaborn as sns
        
       
    _, alfa_X = AA.transform(X)
    
    labels = ['A' + str(i+1) for i in range(AA.n_archetypes)]
    angles = np.linspace(0, 2*np.pi, AA.n_archetypes, endpoint = False)
    angles = np.concatenate((angles, [angles[0]])) 
    
    fig=plt.figure(figsize=(4,4))
    alfa_X = np.concatenate((alfa_X, [alfa_X[0]]))           
    
    ax = fig.add_subplot(111, polar=True)
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, alfa_X, 'o-', markersize=5, linewidth=1.5, color='#273746')
    ax.fill(angles, alfa_X, alpha=0.25)
    ax.set_thetagrids(np.array(angles) * 180.0/np.pi, labels)
    ax.set_title(Title)
    ax.grid(True)
    ax.set_rlim(0,1)
    ax.set_facecolor('#EAECEE')
    
    return ax


def create_simplex_ax(n_archetypes, grid_on = True, gridcolor = '#EAECEE',
                     bordercolor='#4A235A', fontcolor = '#6C3483', fig_size=[10,10]):
        
        """
        # group_color = None, color = None, marker = None, size = None
        group_color:    
            
            Dimension:      n_data x 1
            
            Description:    Contains the category of data point.
        """        
#         plt.style.use('seaborn-paper')
                
#        fig = plt.figure(figsize=[16,8])
        fig, ax = plt.subplots(figsize=fig_size)       
        
        labels = ('A'+str(i + 1) for i in range(n_archetypes))
        rotate_labels=True
        label_offset=0.10
        scaling = False
        sides=n_archetypes
        
        basis = np.array(
                    [
                        [
                            np.cos(2*_*pi/sides + 90*pi/180),
                            np.sin(2*_*pi/sides + 90*pi/180)
                        ] 
                        for _ in range(sides)
                    ]
                )   
    
        for i,l in enumerate(labels):
            if i >= sides:
                break
            x = basis[i,0]
            y = basis[i,1]
            if rotate_labels:
                angle = 180*np.arctan(y/x)/pi + 90
                if angle > 90 and angle <= 270:
                    angle = angle = (angle + 180) % 360 # mod(angle + 180,360)
            else:
                angle = 0
            ax.text(
                    x*(1 + label_offset),
                    y*(1 + label_offset),
                    l,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=angle,
                    fontsize = 12,
                    color = fontcolor
                )
    
        # Clear normal matplotlib axes graphics.
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_frame_on(False)
        
        
        # Plot Grids
        lst_ax_0 = []
        lst_ax_1 = []
        ignore = False
        for i in range(sides):
            for j in range(i + 2, sides):
                if (i == 0 & j == sides):
                    ignore = True
                else:
                    ignore = False                        
    #                
                if not (ignore):                    
                    lst_ax_0.append(basis[i,0] + [0,])
                    lst_ax_1.append(basis[i,1] + [0,])
                    lst_ax_0.append(basis[j,0] + [0,])
                    lst_ax_1.append(basis[j,1] + [0,])
    

        
        ax.plot(lst_ax_0,lst_ax_1, color=gridcolor,linewidth=0.5, alpha = 0.5, zorder=1)
        
        # Plot border
        lst_ax_0 = []
        lst_ax_1 = []
        for _ in range(sides):
            lst_ax_0.append(basis[_,0] + [0,])
            lst_ax_1.append(basis[_,1] + [0,])
    
        lst_ax_0.append(basis[0,0] + [0,])
        lst_ax_1.append(basis[0,1] + [0,])

    #    
        ax.plot(lst_ax_0,lst_ax_1,linewidth=1.5, color=bordercolor, zorder=2) #, **edge_args )               
    
        return fig    


def map_alfa_to_simplex(alfa):
    
    """
    alfa:    2D-array (n_archetypes x n_data)
    """
    
    n_archetypes = alfa.shape[0]
    basis = np.array(
                        [
                            [
                                np.cos(2*_*pi/n_archetypes + 90*pi/180),
                                np.sin(2*_*pi/n_archetypes + 90*pi/180)
                            ] 
                            for _ in range(n_archetypes)
                        ]
                    )
    mapped_alfa = np.dot(alfa.T,basis)
    
    return mapped_alfa
















       


