### Script to hold tests for development of social dataset class. These are bigger tests that might not necessarily fit in a unit testing framework. 
import sys
sys.path.append('../')
from Social_Dataset_Class_v2 import social_dataset
from Social_Dataset_BB import ind_to_dict
import numpy as np
import joblib
from datetime import date

## Test to cross-validate timepoint and interval based implementation of classify_ps. 
reprocess = 0
tag = date.today().isoformat() 
if len(sys.argv) >1:
    reprocess = int(sys.argv[1])
if len(sys.argv) >2:
    tag = sys.argv[2]



if reprocess == 1: 
    ## Load in raw data:
    path = '../../Downloads/V118_03182018_cohousing-Camera-1-Animal-1-Segment1cropped_part'
    scorer0 = 'DeepCut_resnet50_social_v4Feb10shuffle1_100000.h5'
    scorer1 = 'DeepCut_resnet50_social_v4Feb10shuffle1_150000.h5'
    scorers = [scorer0,scorer1]
    scorerindex = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0]
    # scorer = 'DeepCut_resnet50_social_v3Nov2shuffle1_1030000.h5'
    all_datasets = [social_dataset(path+str(i)+scorers[scorerindex[i]],vers = 1) for i in range(30)]
    [dataset.segment_full('training_data/CollectedData_Taiga.h5') for dataset in all_datasets] 
    joblib.dump(all_datasets,'data/seg_traces'+tag)
else:
    all_datasets = joblib.load('data/seg_traces'+tag)
print(len(all_datasets[0].allowed_index_full[0][:,0]))
## We will start just looking at a single case we know to be interesting: 
for i in range(len(all_datasets)):
    for p in range(5):
        current_dataset = all_datasets[i]
        segdict,vsegs_reconstructed,msegs_reconstructed = ind_to_dict(current_dataset.allowed_index_full,p)
        curr_v = 0
        curr_m = 0
        curr_vals = [curr_v,curr_m]
        for i in range(len(segdict)):
            entry = segdict[i]
            curr_vals[entry[-1]] = i 
            print(curr_vals)
        out = current_dataset.classify_v4_ps(p,np.array([np.log(50),3.09,3.09,40.9,40.9]))
        testout = current_dataset.classify_ps_interval(p,np.array([np.log(50),3.09,3.09,40.9,40.9]))
        assert np.all(out == testout)

