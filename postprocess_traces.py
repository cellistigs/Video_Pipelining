import sys
import os
import csv
import numpy as np
from Social_Dataset_Class_v2 import social_dataset
from training_data.Training_Data_Class import training_dataset
import joblib

if __name__ == "__main__":
    datadirectory = sys.argv[1]
    savedirectory = sys.argv[2]
    ## First get the training statistics: 
    path = '../DeepLabCut/pose-tensorflow/models/UnaugmentedDataSet_social_carceaAug29/data-social_carcea'
    trainobj = training_dataset(os.path.join(path,'CollectedData_Taiga.h5'),'')
    vstats,mstats = trainobj.stats_all()

    ## Now get all trace files in the video folder: 
    files = os.listdir(datadirectory)
    traces = [f for f in files if '.h5' in f]
    for tracefile in traces: 
        name = tracefile.split('.h5')[0]
        print(name,'name')
        social_obj = social_dataset(os.path.join(datadirectory,tracefile),vers = 1)
        trajectories = social_obj.render_trajectories(to_render = [0,1,2,3,4,5,6,7,8,9])
        joblib.dump(trajectories,os.path.join(savedirectory,name+'raw'))
        social_obj.filter_full(vstats,mstats)
        filttrajectories = social_obj.render_trajectories(to_render = [0,1,2,3,4,5,6,7,8,9])
        filttrajectories_full = np.concatenate(filttrajectories,axis = 1)
        np.savetxt(os.path.join(savedirectory,name+"filt"),filttrajectories_full,delimiter = ',')     
        #joblib.dump(trajectories,os.path.join(savedirectory,name+'filt'))
        #social_obj.full_ethogram(save = True,show = False,savepath = name+'/')
      
        
    
