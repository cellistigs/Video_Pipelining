import sys
import os
import csv
import datetime
import numpy as np
from Social_Dataset_Class_v2 import social_dataset_online
from training_data.Training_Data_Class import training_dataset
from scipy.io import savemat
import joblib
import re
import yaml

if __name__ == "__main__":
    datadirectory = sys.argv[1]
    savedirectory = sys.argv[2]
    configpath = sys.argv[3]

    ## First get the training statistics: 
    trainpath = '../DeepLabCut/pose-tensorflow/models/UnaugmentedDataSet_social_carceaAug29/data-social_carcea/CollectedData_Taiga.h5'

    ## Next get the config stuff
    config = yaml.load(open(configpath))
    boxcoords = config['coordinates']
    nestcoords = config['nests']

    ## Now get all trace files in the video folder: 
    files = os.listdir(datadirectory)
    traces = [f for f in files if '.h5' in f]
    for tracefile in traces: 
        try: 
            name = tracefile.split('.h5')[0]
            print(name,'name')
            ## We want to do some name handling here. 
            ## Get the roi and time index of each cropped video. 
            sind = re.findall(r"roi_(\d+)",name)
            tind = re.findall(r"cropped_part(\d+)",name)

            assert len(sind) == 1, "We expect only one index per video"
            assert len(tind) == 1, "We expect only one index per video"

            ## Now get the relevant coordinates in the box and the nest: 
            nestselect = nestcoords["box{}".format(sind[0])]
            boxselect = boxcoords["box{}".format(sind[0])]

            ## Get the relative position of the nest coordinates by subtracting off the top left corner of the box:
            nestrel = {}
            for coord in ["x","y"]:
                for ext in ["0","1"]:
                    anchor = boxselect[coord+"0"]
                    selectkey = "{}n{}".format(coord,ext)
                    relpoint = nestselect[selectkey] - anchor
                    nestrel[selectkey] = relpoint

            ## Now we can give these dictionaries to the social dataset object. 
        
            ## Render the social dataset object (use the new online one, which has the set_nest method implemented)
            social_obj = social_dataset_online(os.path.join(datadirectory,tracefile),vers = 1)
            social_obj.set_nest(nestrel)
            social_obj.filter_full_new(trainpath)


            ## Now get trajectories: 
            trajectories = social_obj.render_trajectories(to_render = [i for i in range(10)])
            ## Get velocities:
            velocities = [np.diff(traj,axis = 0) for traj in trajectories]

            ## Get distance traveled: 
            centnorm = [np.linalg.norm(centvel,axis = 1) for centvel in velocities]

            dist_traveled = [np.cumsum(centnorm[cn]) for cn in [0,5]]
            
            ## Get ethograms:
            nest_ethos = [social_obj.nest_ethogram(m) for m in range(2)]
            pursuit_etho = social_obj.shepherding_ethogram()
            ### Additionally get metrics on who is pursuing who. 
            pursuit_premetric,direction= social_obj.orderedtracking()
            ## Extract out the indices where pursuit happened: 
            pursuit_indices = np.where(pursuit_etho)[0]
            ## We can initialize the frame count via the time index parameter. 
            ## TODO: share parameter for video length,frames per second w chunking with the video preprocessing. 
            sec_init = int(tind[0]) * 2400
            pursuit_frames = pursuit_indices
            pursuit_secs = pursuit_frames/30 + sec_init 
            pursuit_time = [str(datetime.timedelta(seconds=float(event))) for event in pursuit_secs]

            pos_lbs = ["tip","leftear","rightear","centroid","tailbase"]
            ind_lbs = ["virgin","dam"]
            all_dict = {} 
            for m,mouse in enumerate(ind_lbs):
                for p,pos in enumerate(pos_lbs):
                    partind = m*5+p
                    ## Added February 5th: 
                    valid_locs, valid_times = social_obj.render_trajectory_valid(partind)
                    ## Initialize array: 
                    empty_array = np.nan*np.zeros((social_obj.dataset[social_obj.scorer].values.shape[0],2))
                    print(valid_times.shape,empty_array.shape)
                    empty_array[valid_times[:,0],:] = valid_locs
                    all_dict["{}_{}_trajraw".format(mouse,pos)] = empty_array 
                    #######################
                    all_dict["{}_{}_traj".format(mouse,pos)] = trajectories[partind]
                    all_dict["{}_{}_vel".format(mouse,pos)] = velocities[partind]
                all_dict["{}_dist".format(mouse)] = dist_traveled[m]
                all_dict["{}_nest".format(mouse)] = nest_ethos[m]
            all_dict["pursuit_bool"] = pursuit_etho
            all_dict["pursuit_direction"] = direction 
            all_dict["pursuit_times"] = pursuit_time

            savemat(os.path.join(savedirectory,name+'processed.mat'),all_dict)
            #trajectories = social_obj.render_trajectories(to_render = [0,1,2,3,4,5,6,7,8,9])
            #joblib.dump(trajectories,os.path.join(savedirectory,name+'raw'))
            #filttrajectories = social_obj.render_trajectories(to_render = [0,1,2,3,4,5,6,7,8,9])
            #filttrajectories_full = np.concatenate(filttrajectories,axis = 1)
            #np.savetxt(os.path.join(savedirectory,name+"filt"),filttrajectories_full,delimiter = ',')     
            ##joblib.dump(trajectories,os.path.join(savedirectory,name+'filt'))
            ##social_obj.full_ethogram(save = True,show = False,savepath = name+'/')
        except Exception as e: 
            print('Encountered exception '+str(e) +' while processing file '+str(tracefile))
            raise NotImplementedError("Debugging")
      
        
    
