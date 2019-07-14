import numpy as np
## Utility functions for Social Dataset
## Takes two sets of segments and returns two lists: one that gives all breakpoints, and another that tells which animals have data between the breakpoints. 
#TODO: TREAT THE ENDS CORRECTLY: extrapolate out to the same length, and prevent cutoff of the last entry. 
def order_segsets(segseta,segsetb):
    segsets = [segseta,segsetb]
    inds_tagged = []
    for s,segset in enumerate(segsets):
        ## First unpack both segment sets
        ind_unpacked = [entry for tup in segset for entry in tup]
        ## Now associate a mouse code with them in the form of a one hot vector. 
        ind_id = np.zeros(len(segsets)) 
        ind_tagged = [(ind_entry,s) for ind_entry in ind_unpacked]
        inds_tagged.append(ind_tagged)
    ## Now sort according to the segment ids. 
    inds_together = sorted(inds_tagged[0]+inds_tagged[1])
    ## unzip, and take a cumulative sum over the identity array that results. 
    inds,ids = zip(*inds_together)
    ## build a onehot array out of ids. 
    template = np.eye(len(segsets))
    ids_array = np.array([template[i] for i in ids])
    intervals = [(inds[i],inds[i+1]) for i in range(len(inds)-1)]
    ids_sorted = np.cumsum(np.array(ids_array),axis = 0)%2
    ## identify repeats, and remove them in the intervals and add the corresponding entries in the ids.
    intervals_trimmed = []
    ids_trimmed = []
    for i,interval in enumerate(intervals):
        if interval[0] == interval[1]:
            pass
        else:
            intervals_trimmed.append(interval)
            ids_trimmed.append(ids_sorted[i])
            print(interval)
    return intervals_trimmed,ids_trimmed

## Make a segment dictionary that splits on the starts and ends of either trajectories.  

def find_segments(indices):
    differences = np.diff(indices)
    all_intervals = []
    ## Initialize with the first element added:
    interval = []
    interval.append(indices[0])
    for i,diff in enumerate(differences):
        if diff == 1:
            pass # interval not yet over
        else:
            # last interval ended
            if interval[0] == indices[i]:
                interval.append(indices[i]+1)
            else:
                interval.append(indices[i]+1)
            all_intervals.append(interval)
            # start new interval
            interval = [indices[i+1]]
        if i == len(differences)-1:
            interval.append(indices[-1]+1)
            all_intervals.append(interval)
    return all_intervals

def order_segsets(segseta,segsetb):
    segsets = [segseta,segsetb]
    inds_tagged = []
    for s,segset in enumerate(segsets):
        ## First unpack both segment sets
        ind_unpacked = [entry for tup in segset for entry in tup]
        ## Now associate a mouse code with them in the form of a one hot vector. 
        ind_id = np.zeros(len(segsets)) 
        ind_tagged = [(ind_entry,s) for ind_entry in ind_unpacked]
        inds_tagged.append(ind_tagged)
    ## Now sort according to the segment ids. 
    inds_together = sorted(inds_tagged[0]+inds_tagged[1])
    ## unzip, and take a cumulative sum over the identity array that results. 
    inds,ids = zip(*inds_together)
    ## build a onehot array out of ids. 
    template = np.eye(len(segsets))
    ids_array = np.array([template[i] for i in ids])
    intervals = [(inds[i],inds[i+1]) for i in range(len(inds)-1)]
    ids_sorted = np.cumsum(np.array(ids_array),axis = 0)%2
    ## identify repeats, and remove them in the intervals and add the corresponding entries in the ids.
    intervals_trimmed = []
    ids_trimmed = []
    for i,interval in enumerate(intervals):
        if interval[0] == interval[1]:
            pass
        else:
            intervals_trimmed.append(interval)
            ids_trimmed.append(ids_sorted[i])
    return intervals_trimmed,ids_trimmed


## Form this array in a few steps. 1. take your set of intervals, and append to each start and end an identity marker. then just argsort all of the indices, and return the sorted timestamps and the sorted id markers. do binary cumsums over the id markers to keep track of when we are in one interval or the other. 
def ind_to_dict_split_for_config(indices,nb_part):
    v_partinds,m_partinds = indices[nb_part],indices[nb_part+5]
    vsegs,msegs = find_segments(v_partinds[:,0]),find_segments(m_partinds[:,0])
    intervals,ids_trimmed = order_segsets(vsegs,msegs)
    processed =np.array(intervals)
    mask = np.array(ids_trimmed)
    
    ## We code each segement for each animal by an element of {0,1,-1}:
    ## 0 = forget
    ## 1 = keep
    ## -1 = switch
    return processed,mask

def ind_to_dict_split(indices,nb_part):
    v_partinds,m_partinds = indices[nb_part],indices[nb_part+5]
    vsegs,msegs = find_segments(v_partinds[:,0]),find_segments(m_partinds[:,0])
    intervals,ids_trimmed = order_segsets(vsegs,msegs)
    processed =np.array(intervals)
    mask = np.array(ids_trimmed)
    ## find points where neither mouse has information: 
    to_forget = np.where(np.sum(mask,axis = 1)!=0)
    processed_rel = processed[to_forget]
    mask_rel = mask[to_forget]
    
    ## We code each segement for each animal by an element of {0,1,-1}:
    ## 0 = forget
    ## 1 = keep
    ## -1 = switch
    return processed_rel,mask_rel

def val_dist(trajraw,intervals,mask,currind,currid,mouseind,mouseid):
    '''
    A function to calculate distances betweeen the end of one trajectory and the beginning of another. Needs a mask argument because it needs to think about validity of all possible trajectories!!!
    '''
    
    if mask[currind,currid] == 0:
        return np.nan
    else:
        ## We don't have to worry about trajectories not existing because we should only be seeing processed ones in both arguments. 
        end = intervals[mouseind][-1]-1 ## -1 for interval indexing. 
        start = intervals[currind][0]
        mouseid = int(mouseid)
        trajstart = trajraw[start,currid*2:currid*2+2]
        trajend = trajraw[end,mouseid*2:mouseid*2+2]
        distance = np.linalg.norm(trajstart-trajend)
        return distance
    
def val_time(intervals,mask,currind,currid,mouseind,mouseid):
    if mask[currind,currid] == 0:
        return np.nan
    else:
        ## We don't have to worry about trajectories not existing because we should only be seeing processed ones in both arguments. 
        end = intervals[mouseind][-1]-1
        start = intervals[currind][0]
        return start-end 

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
