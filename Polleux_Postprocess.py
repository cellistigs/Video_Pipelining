from Social_Dataset_Class_v2 import social_dataset
import pandas as pd
from scipy.io import savemat
import sys
import os
import numpy as np

if __name__ == "__main__":
    path = sys.argv[1]
    dataset = pd.read_hdf(path)
    ## First get the centroid positions. 
    centroid = dataset["DeepCut_resnet50_fearcond_polleuxSep17shuffle1_1030000"]["centroid"][["x","y"]].values
    ## Now get the velocities: 
    centvel = np.diff(centroid,axis = 0)
    ## Now get the distance traveled: 
    centnorm = np.linalg.norm(centvel,axis = 1)
    centcumsum = np.cumsum(centnorm)
    ## Now turn this into a dictionary: 
    all_dict = {"position":centroid,"velocity":centvel,"distance_traveled":centcumsum}
    ## Now save as a matrix
    savemat(path.split(".h5")[0]+"processed",all_dict)
    
    
    
    
