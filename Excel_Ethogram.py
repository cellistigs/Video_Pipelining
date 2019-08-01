import numpy as np
import sys
import os
import joblib
import pandas as pd
from Social_Dataset_utils import filepaths,datapaths,excelpaths


if __name__ == "__main__":
    folderpath = sys.argv[1]
    unique_string = sys.argv[2].split('cropped_part')[0]
    sheet_tag = excelpaths(folderpath)[0]
    dataset_paths = datapaths(folderpath)
    file_paths = datapaths(folderpath) 
    ## Unique identifier to id the ones that we care about:
    ## Annoying: First get the number and positions of all datasets:

    #numbers = [int(d.split('cropped_part')[-1].split('DeepCut')[0]) for d in file_paths if unique_string in d and 'ethogram' not in d]
    #max_ind = 47#np.max(numbers)
    max_ind = 36#len(dataset_paths)-1 

    ## Behavior excel spreadsheet name: 
    #sheet_tag = sys.argv[2]
    ## Write out some important strings: 
    behavior_tag = 'Behavior'
    start_tag = "Start (s)"
    stop_tag = "Stop (s)"

    dam_pos_tag = "Mother in nest"
    virg_pos_tag = "Virgin in rest"
    pursuit_tag = "Mom agressing"

    ## Get the spreadsheet: 
    excel_data = pd.read_excel(sheet_tag)

    ## Just get out the parts we care about: 
    trimmed_data = excel_data[[behavior_tag,start_tag,stop_tag]]

    ## Further separate out into the behaviors we care about 
    dam_pos = trimmed_data[trimmed_data[behavior_tag] == dam_pos_tag][[start_tag,stop_tag]]
    virg_pos = trimmed_data[trimmed_data[behavior_tag] == virg_pos_tag][[start_tag,stop_tag]]
    pursuit = trimmed_data[trimmed_data[behavior_tag] == pursuit_tag][[start_tag,stop_tag]]
    
    ## Now package up the starts and stops : 
    ethogram_sources = [dam_pos,virg_pos,pursuit]
    ethogram_name = ['full_mother_nest_','full_virgin_nest_','full_pursuit_']
    for s,source in enumerate(ethogram_sources):
        ## initialize ethogram: 
        ethogram  = np.zeros((1+max_ind)*36000,)
        for ind,ent in source.iterrows():
            start,end = np.round(30*ent[start_tag]).astype(int),np.round(30*ent[stop_tag]).astype(int)
            ethogram[start:end] = 1
            
        namestring = folderpath+'/'+'dataset_'+unique_string+ethogram_name[s]+'gt_ethogram'
        joblib.dump(ethogram,namestring)

    
