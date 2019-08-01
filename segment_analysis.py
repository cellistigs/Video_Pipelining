## Script for automatic analysis post dlc tracking. 
from Social_Dataset_Class_v2 import social_dataset
from training_data.Training_Data_Class import training_dataset
import joblib
import moviepy
import os
import sys
sys.path.insert(0,'../auxvolume/temp_videofolder')
from config import xn,yn

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
    traindata = training_dataset(trainingdatapath+'CollectedData_Taiga.h5',trainingdatapath)
    vstats,mstats = [traindata.stats_wholemouse([i+1 for i in range(100)],m) for m in range(2)] 
    ## Load in all relevant files
    folderpath = '../auxvolume/temp_videofolder' 
    datafiles = filepaths(folderpath)
    ## Load in the nest information from the config file: 
    print(xn,yn,"can do")
    ## Analyze with accompanying movies (parallelize?)
    moviefiles = [moviepath(filepath) for filepath in datafiles] 
    datatuple = zip(datafiles,moviefiles)
    datasets = []
    for i,datatup in enumerate(datatuple):
        filename_local = datatup[0].split('.')[-2].split('/')[-1]
        try:
            ## Load Datasets 
            dataset = social_dataset(datatup[0],vers= 1)
            ## Import Movies
            dataset.import_movie(datatup[1])
            dataset.bounds = (xn,yn)
            ## Clean data
            dataset.filter_full(vstats,mstats)
            ## Save data: 
            ## REMOVE MOVIE because it doesn't play well with joblib. 
            dataset.movie = ['removed for saving']
            joblib.dump(dataset,'../auxvolume/temp_videofolder/dataset_'+filename_local)
            ## Find shepherding events. 
            dataset.full_ethogram(show = False,save = True,savepath = '../auxvolume/temp_videofolder/') 
            print('saved at'+'../auxvolume/temp_videofolder/dataset_'+filename_local)
        except:
            print('Data Not successfully loaded for '+filename_local)
