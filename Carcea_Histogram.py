import sys
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import numpy as np
from Social_Dataset_Class_v2 import social_dataset_online
from training_data.Training_Data_Class import training_dataset
from scipy.io import loadmat
import joblib
import re
import yaml

if __name__ == "__main__":
    datadirectory = sys.argv[1]
    savedirectory = sys.argv[2]
    configpath = sys.argv[3]

    ## Next get the config stuff
    config = yaml.load(open(configpath))
    boxcoords = config['coordinates']
    nestcoords = config['nests']

    ## Now get all trace files in the video folder: 
    files = os.listdir(datadirectory)
    traces = [f for f in files if '.mat' in f]
    for tracefile in traces: 
        try: 
            name = tracefile.split('.mat')[0]
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
            exts = []
            for coord in ["x","y"]:
                extent = boxselect[coord+'1']-boxselect[coord+'0']
                exts.append(extent)
            #    for ext in ["0","1"]:
                    #anchor = boxselect[coord+"0"]
                    #selectkey = "{}n{}".format(coord,ext)
        except Exception as e: 
            print('Encountered exception '+str(e) +' while processing file '+str(tracefile))
            raise NotImplementedError("Debugging")
        print(exts,"extensions")
        ## Now load in the actual data. 
        data = loadmat(os.path.join(datadirectory,tracefile))
        ## Now make a histogram for the mother and the virgin. 
        for i in ["dam","virgin"]:
            traces = data["{}_centroid_traj".format(i)]
            bins = [np.arange(0,exts[0]+10,exts[0]/10),np.arange(exts[1],-10,-10)]
            H,xaxis,yaxis,im = plt.hist2d(traces[:,0],traces[:,1],bins = 50)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.savefig(os.path.join(savedirectory,name+"_{}_hist".format(i)))
            plt.close()



