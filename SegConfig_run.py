from SegConfig import Configuration,Optimizer
from Social_Dataset_utils import *
import joblib

if __name__ == '__main__':
    
    ## Import a preprocessed dataset so we can work quickly. 
    datapath = 'data/all_data_finetuned_votes'
    all_datasets = joblib.load(datapath) # Contains all 30 days, already preprocessed. 
    ## We will work with one to begin with:
    for dataset in [all_datasets[19]]:
        indices = dataset.allowed_index_full
        for nb_part in [0]:
            intervals,mask= ind_to_dict_split(indices,nb_part)
        
            ## Now render the appropriate trajectory from a signature+ segment dictionary
            vraw,mraw = dataset.select_trajectory(nb_part),dataset.select_trajectory(nb_part+5)
            trajraw = np.concatenate((vraw,mraw),axis=1)
            intervals,mask= ind_to_dict_split_for_config(indices,nb_part)
            end = 100
            intervals = intervals[:end]
            mask = mask[:end]
            trajraw = trajraw[:intervals[-1,-1],:]
            conf = Configuration(trajraw,intervals,mask)
            opt = Optimizer(conf,'left')
            #conf.plot_full_cost(segindex = [600,650])
            opt.optimize0(2)

