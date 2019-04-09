## Script for automatic analysis post dlc tracking. 
from Social_Dataset_Class_v2 import social_dataset
from training_data.Training_Data_Class import training_dataset
import moviepy
import os
import sys
def filepaths(folderpath):
    ## Look for all files that we should analyze within the folder: 
    ## These will be distinguished by the name cropped_part*.mp4
    all_files = os.listdir(folderpath)
    data = [folderpath+'/'+fileset for fileset in all_files if fileset.split('.')[-1] == 'h5' and 'cropped_part' in fileset.split('.')[-2]]
    return data 
def moviepath(filepath):
    relevant_part = filepath.split('DeepCut')[0]
    movie_append = '.mp4'
    return relevant_part+movie_append

if __name__ == '__main__':
    ## Load in the training data: 
    trainingdatapath = "../DeepLabCut/pose-tensorflow/models/UnaugmentedDataSet_social_v3Nov2/data-social_v3/"
    traindata = training_dataset1(trainingdatapath+'CollectedData_Taiga.h5',trainingdatapath)
    vstats,mstats = [traindata.stats_wholemouse([i+1 for i in range(100)],m) for m in range(2)] 
    ## Load in all relevant files
    folderpath = sys.argv[1]
    datafiles = filepaths(folderpath)
    ## Load in the nest information from the config file: 
    sys.path.insert(0,folderpath)
    ## Gross but less gross than other things
    from config import xn,yn
    ## Analyze with accompanying movies (parallelize?)
    moviefiles = [moviepath(filepath) for filepath in datafiles] 
    datatuple = zip(datafiles,moviefiles)
    datasets = []
    for i,datatup in enumerate(datatuple):
        ## Load Datasets 
        dataset = social_dataset(datatup[0])
        ## Import Movies
        dataset.import_movie(datatup[1])
        dataset.bounds = (xn,yn)
        ## Clean data
        dataset.filter_full(vstats,mstats)
        ## Find shepherding events. 
        dataset.full_ethogram() 
