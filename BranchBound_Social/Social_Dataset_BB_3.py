### Code to evaluate the distribution of intervals in the data. 
import sys
sys.path.append('../')
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from Social_Dataset_BB_2 import *

## First import data: 
def process_data(datapath,to_return = None,parts = None):
    ## Import a preprocessed dataset so we can work quickly.
    all_datasets = joblib.load(datapath) # Contains all 30 days, already preprocessed.
    ## We will work with one to begin with:
    dataset_array = []
    for nb_dataset in to_return:
        dataset = all_datasets[nb_dataset]
        indices = dataset.allowed_index_full
        part_array = []
        for nb_part in parts:
            ## Now render the appropriate trajectory from a signature+ segment dictionary
            vraw,mraw = dataset.select_trajectory(nb_part),dataset.select_trajectory(nb_part+5)
            trajraw = np.concatenate((vraw,mraw),axis=1)
            intervals,mask= ind_to_dict_split(indices,nb_part)
            part_array.append([trajraw,intervals,mask])
        dataset_array.append(part_array)
    return dataset_array 

def process_datafake_lin(length,goodpoints,switchpoints):
    traj,indices = fake_data_switch(length,goodpoints,switchpoints)
    intervals,mask = ind_to_dict_split(indices,0)
    return [[[traj/np.sqrt(2),intervals,mask]]]

def process_datafake(length,goodpoints,switchpoints):
    traj,indices = fake_data_adv(length,goodpoints,switchpoints)
    intervals,mask = ind_to_dict_split(indices,0)
    return [[[traj,intervals,mask]]]


### To process intra-segment distances. 
def intra_tv(trajraw,intervals,mask,i,m):
    if mask[i,m] == 0:
        return np.nan
    else:
        interval = intervals[i,:]
        # Isolate traces at relevant times: 
        traj = trajraw[slice(*interval),tuple(m*2+np.array([0,1]))]
        tv = np.sum(np.linalg.norm(np.diff(traj,axis = 0),axis = 1))
        #print(traj,np.diff(traj,axis=0),np.linalg.norm(np.diff(traj,axis = 0),axis = 1))
        return tv 

### To process intra-segment distances. 
def intra_dist(trajraw,intervals,mask,i,m):
    if mask[i,m] == 0:
        return np.nan
    else:
        interval = intervals[i,:]
        # Isolate traces at relevant times: 
        end = trajraw[interval[-1]-1,m*2+np.array([0,1])]
        start = trajraw[interval[0],m*2+np.array([0,1])]
        return np.linalg.norm(end-start)

def intra_time(intervals,mask,i,m):
    if mask[i,m] == 0:
        return np.nan
    else:
        interval = intervals[i,:]
        return interval[-1]-interval[0]

### To process inter-segment distances. 
def local_inter_rates(trajraw,intervals,mask,i,last_valid_ind,last_valid_ids):
    dists = [val_dist(trajraw,intervals,mask,i,m,last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
    times = [val_time(intervals,mask,i,m,last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
    rates = [dists[m]/times[m] for m in range(2)]
    return rates,dists,times 

### To process inter-segment distances. 
def local_inter_rates_project(trajraw,intervals,mask,i,next_valid_ind,next_valid_ids):
    dists = [val_dist(trajraw,intervals,mask,i,m,next_valid_ind[m],next_valid_ids[m]) for m in range(2)]
    times = [abs(val_time(intervals,mask,i,m,next_valid_ind[m],next_valid_ids[m])) for m in range(2)]
    rates = [dists[m]/times[m] for m in range(2)]
    return rates,dists,times 

def local_inter_rates_switch(trajraw,intervals,mask,i,last_valid_ind,last_valid_ids):
    dists = [val_dist(trajraw,intervals,mask,i,abs(1-m),last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
    times = [val_time(intervals,mask,i,abs(1-m),last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
    rates = [dists[m]/times[m] for m in range(2)]
    return rates,dists,times 

def local_inter_rates_project_switch(trajraw,intervals,mask,i,next_valid_ind,next_valid_ids):
    print(intervals.shape,next_valid_ind,next_valid_ids,i)
    dists = [val_dist(trajraw,intervals,mask,i,abs(1-m),next_valid_ind[m],next_valid_ids[m]) for m in range(2)]
    times = [abs(val_time(intervals,mask,i,abs(1-m),next_valid_ind[m],next_valid_ids[m])) for m in range(2)]
    rates = [dists[m]/times[m] for m in range(2)]
    return rates,dists,times 

def local_intra_rates(trajraw,intervals,mask,i):
    dists = [intra_dist(trajraw,intervals,mask,i,m) for m in range(2)]
    times = [intra_time(intervals,mask,i,m) for m in range(2)]
    rates = [dists[m]/times[m] for m in range(2)]
    return rates,dists,times 

def local_intra_tv_rates(trajraw,intervals,mask,i):
    dists = [intra_tv(trajraw,intervals,mask,i,m) for m in range(2)]
    times = [intra_time(intervals,mask,i,m) for m in range(2)]
    tv = [dists[m]/times[m] for m in range(2)]
    return tv,dists,times

## To process intra-segment distances. 
def local_intra_dists(trajraw,intervals,mask,i):
    dists = [intra_dist(trajraw,intervals,mask,i,m) for m in range(2)]
    return dists

def local_intra_times(intervals,mask,i):
    times = [intra_time(intervals,mask,i,m) for m in range(2)]
    return times 


def local_intra_tv_dist(trajraw,intervals,mask,i):
    dists = [intra_tv(trajraw,intervals,mask,i,m) for m in range(2)]
    return dists


## Now we will calculate the histogram of inter-interval distances that are found in the raw data. 
def calculate_inter_rates(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the inter-interval distances. 
    for id_ind in np.arange(1,len(id_array)):
        last_valid_ind,last_valid_ids = last_valid(id_array[:id_ind,:])
        rates,_,_ = local_inter_rates(trajraw,intervals,mask,id_ind,last_valid_ind,last_valid_ids)
        rates_array[id_ind,:] = np.array(rates)
        ## we get nans *only* from 0/0. 
    return rates_array

def calculate_inter_rates_gap(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the inter-interval distances. 
    ## For the version with the gap we shift the interval index one forwards, so that we are always calculating the distance to the 
    ## last valid index that does not take into account the index that we are acutally registered to. 
    for id_ind in np.arange(1,len(id_array)-1):
        last_valid_ind,last_valid_ids = last_valid(id_array[:id_ind,:])
        rates,_,_ = local_inter_rates(trajraw,intervals,mask,id_ind+1,last_valid_ind,last_valid_ids)
        rates_array[id_ind,:] = np.array(rates)
        ## we get nans *only* from 0/0. 
    return rates_array[1:,:]

## Calculate times: 
def calculate_intra_time(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    times_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(len(id_array)):
        ids = id_array[id_ind,:]
        times = [intra_time(intervals,mask,id_ind,m) for m in range(2)]
        times_array[id_ind,:] = np.array(times)
        ## we get nans *only* from 0/0. 
    return times_array 

## Now we will calculate the histogram of intra-interval distances that are found in the segments. 
def calculate_intra_rates(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(len(id_array)):
        ids = id_array[id_ind,:]
        rates,_,_ = local_intra_rates(trajraw,intervals,mask,id_ind)
        rates_array[id_ind,:] = np.array(rates)
        ## we get nans *only* from 0/0. 
    return rates_array

## Similarly, calculate the TV distance traveled as found in the segments.
def calculate_intra_tv(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(len(id_array)):
        ids = id_array[id_ind,:]
        rates,_,_ = local_intra_tv_rates(trajraw,intervals,mask,id_ind)
        rates_array[id_ind,:] = np.array(rates)
        ## we get nans *only* from 0/0. 
    return rates_array

def calculate_intra_diff(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(len(id_array)):
        ids = id_array[id_ind,:]
        tv,_,_ = local_intra_tv_rates(trajraw,intervals,mask,id_ind)
        rates,_,_ = local_intra_rates(trajraw,intervals,mask,id_ind)
        rates_array[id_ind,:] = np.array(tv)-np.array(rates)
        ## we get nans *only* from 0/0. 
    return rates_array

### Specialized functions for our own purposes. 
def calculate_intra_wide(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    print(rates_array.shape)
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(1,len(id_array)-2):
        last_valid_ind,last_valid_ids = last_valid(id_array[:id_ind,:])
        next_valid_ind,next_valid_ids = first_valid(id_array[id_ind+1:,:])
        ## Calculate total distance:
        _,tvdist,tvtime = local_intra_rates(trajraw,intervals,mask,id_ind)
        _,predist,pretime = local_inter_rates(trajraw,intervals,mask,id_ind,last_valid_ind,last_valid_ids)
        _,postdist,posttime = local_inter_rates_project(trajraw,intervals,mask,id_ind,next_valid_ind+id_ind+1,next_valid_ids)
        print(tvdist,predist,postdist,'dists',id_ind)
        dist = np.array(tvdist)+np.array(predist)+np.array(postdist)
        time = np.array(tvtime)+np.array(pretime)+np.array(posttime)
        print(tvtime,pretime,posttime)
        rates_array[id_ind] = dist/time
    return rates_array[1:,:] 

def calculate_wide_diff(part_array,signature):
    baseline = calculate_inter_rates_gap(part_array,signature)
    of_interest = calculate_intra_wide(part_array,signature)    
    return of_interest-baseline

def calculate_straight_cost(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    for id_ind in np.arange(1,len(id_array)-1):
        last_valid_ind,last_valid_ids = last_valid(id_array[:id_ind,:])
        next_valid_ind,next_valid_ids = first_valid(id_array[id_ind+1:,:])
        ## Calculate total distance:
        _,predist,pretime = local_inter_rates(trajraw,intervals,mask,id_ind,last_valid_ind,last_valid_ids)
        _,postdist,posttime = local_inter_rates_project(trajraw,intervals,mask,id_ind,next_valid_ind+id_ind+1,next_valid_ids)
        dist = np.array(predist)+np.array(postdist)
        time =np.array(pretime)+np.array(posttime)
        rates_array[id_ind] = dist/time
    return rates_array[1:,:] 

def calculate_swap_cost(part_array,signature):
    ## First, convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    rates_array = np.zeros(np.shape(id_array))
    ## Now, traverse this id array and find the intra-interval distances. 
    print(len(id_array))
    for id_ind in np.arange(1,len(id_array)-1):
        last_valid_ind,last_valid_ids = last_valid(id_array[:id_ind,:])
        next_valid_ind,next_valid_ids = first_valid(id_array[id_ind+1:,:])
        ## Calculate total distance:
        _,predist,pretime = local_inter_rates_switch(trajraw,intervals,mask,id_ind,last_valid_ind,last_valid_ids)
        _,postdist,posttime = local_inter_rates_project_switch(trajraw,intervals,mask,id_ind,next_valid_ind+id_ind+1,next_valid_ids)
        dist = np.array(predist)+np.array(postdist)
        time =np.array(pretime)+np.array(posttime)
        print(dist,time)
        rates_array[id_ind] = dist/time
    return rates_array[1:,:] 


## Define a plotting function that plots a certain window before and after a segment of choice (indicated by an interval index).  
def plot_sequential(part_array,signature,center,radius = 5):
    ## First convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    intervals_centered = intervals[center-radius:center+radius+1]
    
    ### Indicate the start and end of the figure: 
    first_point,end_point = intervals_centered[0,0],intervals_centered[-1,-1]
    ### Indicate the start and end of the target interval:
    first_center,end_center = intervals[center,0],intervals[center,1]
    print(center,'center')
    ids_centered = id_array[center-radius:center+radius+1]
    print(ids_centered.shape)
    fig,ax = plt.subplots(2,1)
    [ax[j].set_xlim(first_point,end_point) for j in range(2)]
    [ax[j].axvline(x = first_center,color = 'black') for j in range(2)]
    [ax[j].axvline(x = end_center-1,color = 'black') for j in range(2)]
    ax[0].set_title('X coordinate segments for segment: '+str(center))
    ax[1].set_title('Y coordinate segments for segment: '+str(center))
    ax[1].set_xlabel('Time Index')
    ax[0].set_ylabel('coordinate value')
    plt.tight_layout()
    colors = ['blue','red']
    ## First plot visual aids to show where the points of interest are: 
    for index in np.arange(2*radius+1):
        for m in range(2):
            if not np.isnan(ids_centered[index,m]):
                ## Pull out the relevant trajectory: 
                vertslice = slice(*intervals_centered[index,0:2])
                vertslice = slice(*intervals_centered[index,0:2])
                horslice = tuple(int(ids_centered[index,m])*2+np.array([0,1]))
                traj = trajraw[vertslice,horslice]
                x = np.arange(*intervals_centered[index])
                ax[0].plot(x,traj[:,0],color = colors[m])
                ax[1].plot(x,traj[:,1],color = colors[m])
                plt.pause(0.1)
    #plt.show()
    plt.savefig('sequential')
            
## Write down the new version of the cost with calculation of tv norm on segments, with two regularization parameters. 


## Define a plotting function that plots a certain window before and after a segment of choice (indicated by an interval index).  
def plot_cost(part_array,signature,center,):
    ## First convert the signature into an id. 
    trajraw = part_array[0]
    intervals = part_array[1]
    mask = part_array[2]
    id_array = process_signature(signature,mask)
    intervals_centered = intervals[center-radius:center+radius+1]
    ### Indicate the start and end of the figure: 
    first_point,end_point = intervals_centered[0,0],intervals_centered[-1,-1]
    ### Indicate the start and end of the target interval:
    first_center,end_center = intervals[center,0],intervals[center,1]
    ids_centered = id_array[center-radius:center+radius+1]
    fig,ax = plt.subplots(2,1)
    [ax[j].set_xlim(first_point,end_point) for j in range(2)]
    [ax[j].axvline(x = first_center,color = 'black') for j in range(2)]
    [ax[j].axvline(x = end_center-1,color = 'black') for j in range(2)]
    ax[0].set_title('X coordinate segments for segment: '+str(center))
    ax[1].set_title('Y coordinate segments for segment: '+str(center))
    ax[1].set_xlabel('Time Index')
    ax[0].set_ylabel('coordinate value')
    plt.tight_layout()
    colors = ['blue','red']
    ## First plot visual aids to show where the points of interest are: 
    for index in np.arange(2*radius+1):
        for m in range(2):
            if not np.isnan(ids_centered[index,m]):
                ## Pull out the relevant trajectory: 
                vertslice = slice(*intervals_centered[index,0:2])
                vertslice = slice(*intervals_centered[index,0:2])
                horslice = tuple(int(ids_centered[index,m])*2+np.array([0,1]))
                traj = trajraw[vertslice,horslice]
                x = np.arange(*intervals_centered[index])
                ax[0].plot(x,traj[:,0],color = colors[m])
                ax[1].plot(x,traj[:,1],color = colors[m])
                plt.pause(0.1)
    plt.show()



        
interactive = False 
exploratory = True 

if __name__ == '__main__':
    interactive = False

if __name__ == '__main__':
    ## Import a preprocessed dataset so we can work quickly.
    datapath = '../data/all_data_finetuned_votes'
    dataset_nbs = [0]
    part_nbs = [0]
    processed = process_data(datapath,dataset_nbs,part_nbs)
    processed = process_datafake_lin(25,[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,23,24],[12,13,14])
    for dataset_nb in dataset_nbs:
        for part_nb in part_nbs:
            processed_local = processed[dataset_nb][part_nb]
            rv_diff =calculate_wide_diff(processed_local,np.ones(len(processed_local[1])*2))
            cutoff = 5
            outpoints = np.where(rv_diff>cutoff)[0]
            ## Remove nans from 0/0 rates: 
            rates_valid_0 = rv_diff[~np.isnan(rv_diff)].flatten()
            rates_valid = rates_valid_0[np.where(rates_valid_0>0)]
            vals,bins,_ = plt.hist(rates_valid,bins = 100,log= False,density = True)
            plt.title('Distribution of Normalized TV norm (ground truth)')
            plt.xlabel('Normalized TV cost')
            plt.ylabel('density')
            plt.axvline(x = cutoff)
            #plt.plot(bins,lambda_mle*np.exp(-lambda_mle*bins))
            if interactive == True:
                plt.pause(3)
            else:
                plt.show()
            plt.close()
            [plot_sequential(processed_local,np.ones(len(processed_local[1])*2),intpoint-1) for intpoint in outpoints] 
            if exploratory == True:
                [plot_sequential(processed_local,np.ones(len(processed_local[1])*2),intpoint-1) for intpoint in outpoints] 
            ## It seems that a cutoff of around 8 successfully isolates the segments that jump from the real ones. 
            ## Look at how the distribution varies with segment length. 
            times = calculate_intra_time(processed_local,np.ones(len(processed_local[1])*2))[1:,:] 
            time_points = times[~np.isnan(rv_diff)].flatten()
            time_points = time_points[np.where(rates_valid_0>0)]
            plt.plot(rates_valid,time_points,'o',markersize = 1)
            plt.axvline(x = cutoff)
            plt.title('Segment Length vs. Normalized TV norm')
            plt.xlabel('Normalized TV norm')
            plt.ylabel('Segment Length')
            plt.show()
            ## It seems that the correlation with the length of the underlying segment is not too important. 
            ## Fit a distribution to the accepted points. 
            ## Calculate MLE: 
            lambda_mle = 1/np.mean(rates_valid[np.where(rates_valid<cutoff)])
            ## Fit to truncated data to see the resultant distribution. 
            vals,bins,_ = plt.hist(rates_valid,bins = 100,log= False,density = True)
            plt.plot(bins,lambda_mle*np.exp(-lambda_mle*bins))
            plt.title('Distribution of Normalized TV norm (exponential fit with lambda ='+str(lambda_mle)[:4]+')')
            plt.xlabel('Normalized TV cost')
            plt.ylabel('density')
            plt.show()
            ## A probabilistic interpretation: hypothesis testing with a p cutoff. 
            p_cutoff = lambda_mle*np.exp(-lambda_mle*8)
            print('cutoff probability is: '+str(p_cutoff))

            ## We now want a distribution of cost benefit due to switches. 
            ## These switches only incur a cost in terms of the cost of transitioning to them. 
            yess = calculate_straight_cost(processed_local,np.ones((len(processed_local[2]))*2)) 
            argh = calculate_swap_cost(processed_local,np.ones((len(processed_local[1]))*2)) 
            diff = yess-argh

            diff_points = diff[~np.isnan(diff)].flatten()
            diff_clean = diff_points[~np.isinf(diff_points)]
            plt.hist(diff_clean,normed = True,bins = 100) 
            plt.title('Distribution of cost change due to flipping of intervals.')
            plt.xlabel('Cost difference post flip.')
            plt.ylabel('density')
            plt.show()

            ## Correlate this with segment length: 
            time_diff_points = times[~np.isnan(diff)].flatten()
            time_clean = time_diff_points[~np.isinf(diff_points)]
            plt.plot(diff_clean,time_clean,'o',markersize=1)
            plt.title('Differential cost distribution as a function of segment length')
            plt.show()

            ## 
            switchpoints = np.where(diff>0)[0]
            #if exploratory == True:
                #[plot_sequential(processed_local,np.ones(len(processed_local[1])*2),intpoint) for intpoint in switchpoints] 
            
            
        
        
