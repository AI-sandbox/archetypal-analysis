"""
Created on Friday June 11 2021
@author: Julia Gimbernat
This function is developed based on "Archetypal Analysis" by Adele Cutler and Leo
Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347.
It requires the Archetypal Analysis package by Benyamin Motevalli: https://data.csiro.au/collections/collection/CI40600
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from src.archetypal_analysis import *
from src.snp_reader import SNPReader

def run_pipeline(input_file, output_file, n_archetypes, tolerance = 0.001,
                 max_iter = 200, random_state = 0, C = 0.0001,
                 initialize = 'furthest_sum', redundancy_try = 30):
    """
    This function computes the Archetypal Analysis algorithm given a .vcf file or a .npy pca file.
    It outputs the Q and Z files (similar to Q and P files of ADMIXTURE).
    
    
    Parameters:
    -----------
    input_file:     Defines the input file / path. File must be in vcf, bed, pgen or npy format. If npy, assumes data is already projected.
    
    output_file:    Defines the output file / path. File name does not need any extensions.
   
    n_archetypes:   Defines the number of archetypes.
    
    tolerance:      Defines when to stop optimization.
    max_iter:       Defines the maximum number of iterations
    random_state:   Defines the random seed number for initialization. No effect if "furthest_sum" is selected.       
                    
    C:              is a constraint coefficient to ensure that the summation of
                    alfa's and beta's equals to 1. C is considered to be inverse
                    of M^2 in the original paper.
    initialize:     Defines the initialization method to guess initial archetypes:
                        1. furthest_sum (Default): the idea and code taken from https://github.com/ulfaslak/py_pcha and the original author is: Ulf Aslak Jensen.
                        2. random:  Randomly selects archetypes in the feature space. The points could be any point in space
                        3. random_idx:  Randomly selects archetypes from points in the dataset.
    
    
    
    Output:
    -----------
    Q_file: 
        Dimension:  n_individuals x n_archetypes 
        
                    Each row defines the weight coefficients for each 
                    archetype to approximate data point i.
                    
    Z_file:
        Dimension:  n_dim x n_archetypes
        
                    Array of archetypes. Each columns represents an archetype.
    """
    if input_file.endswith('.npy'):
      print('Loading already projected data...')
      pca_result = np.load(input_file)
    else:
      snpreader = SNPReader()
      G = snpreader.read_data(input_file)
      print('Computing PCA...')
      pca = PCA(random_state=random_state)
      pca_result = pca.fit_transform(G)
      save_pca_file = f"{input_file.split('.')[0]}_pca_projection.npy"
      print(f'Saving PCA results to {save_pca_file} to allow reuse...')
      np.save(save_pca_file, pca_result)

    # Perform Archetypal Analysis
    AA = ArchetypalAnalysis(n_archetypes = n_archetypes, 
                            tolerance = tolerance, 
                            max_iter = max_iter, 
                            random_state = random_state, 
                            C = C,
                            initialize = initialize,
                            redundancy_try = redundancy_try)
    print('Fitting AA...')
    AA.fit(pca_result)
    print('Saving Q...')
    # save Q file
    Q = np.transpose(AA.alfa)
    pd.DataFrame(Q).to_csv(output_file+f'.{n_archetypes}.Q', index=False, header=False, sep=' ')
    print('Saving Z...')
    # save Z file
    Z = AA.archetypes
    Z = pca.inverse_transform(AA.archetypes.T)
    print('Z shape is ...', Z.shape)
    pd.DataFrame(Z).to_csv(output_file+f'.{n_archetypes}.Z', index=False, header=False, sep=' ')
    return 0