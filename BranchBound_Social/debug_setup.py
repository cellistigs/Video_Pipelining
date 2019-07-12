## Script to setup the variables you will use most often for debugging. 
from Social_Dataset_BB_testdata import SwitchCircle
from SegConfig import *

length = 1000
segexp = 80
gapexp = 20
switch = 0.8
period = 500

test = SwitchCircle(length,segexp,gapexp,switch,period=period)

conf = Configuration(test.trajectory,test.intervals_trimmed,test.ids_trimmed,0.12566288,0) 

opt = Optimizer(conf,'both')

gt = conf.return_checked(generate_groundtruth(test))

