import sys
sys.path.append('../')
from Social_Dataset_Class_v2 import social_dataset
from scipy.interpolate import interp1d
from Branch_Bound.ops import *
import numpy as np
import joblib

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
    return intervals_trimmed,ids_trimmed


## Form this array in a few steps. 1. take your set of intervals, and append to each start and end an identity marker. then just argsort all of the indices, and return the sorted timestamps and the sorted id markers. do binary cumsums over the id markers to keep track of when we are in one interval or the other. 
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

def id_to_sig(all_ids):
    ## first convert to +-1
    processed = (all_ids*2-1)*np.array([-1,1])
    ## nan -> 0
    processed[np.isnan(processed)] = 0
    ## reshape:
    return processed.flatten()

def process_signature(signature,mask):
    ## Check for parity. 
    if len(signature)%2 == 1:
        signature = np.append(signature,0)
    sig_array = signature.reshape(-1,2)
    length = sig_array.shape[0]
    mask_cut = mask[:length,:]
    ## Filter out nonexistent entries:
    sig_masked = sig_array*mask_cut
    ## Resolve switch conflicts: 
    is_conflict = np.squeeze(abs(np.diff(sig_masked,axis = 1))==2)
    sig_masked[is_conflict] = (sig_masked[is_conflict]+1)//2
    ## Send 0 to nan: 
    sig_masked[sig_masked==0] = np.nan
    ## Multiply to recover correct identities:
    varray = -1*np.ones((length,1))
    marray = np.ones((length,1))
    idarray = np.concatenate((varray,marray),axis = 1)
    all_ids = (sig_masked*idarray+1)//2
    return all_ids

def compile_intervals(all_ids,all_intervals,shift = 0):
    ## We want to be a little careful here. We want to just barely catch the end of the last segment when we have a shift. 
    all_intervals = all_intervals[:len(all_ids),:]
    intervals = []
    ids = []
    coords = []
    is_gen = np.arange(len(all_intervals))>=shift
    ## Pull out the end of the last valid trajectory BEFORE the start of the current one. 
    lv_ind,lv_id = last_valid(all_ids[:shift])
    
    for i in range(2):
        ## Process rest of data: 
        is_id = ~np.isnan(all_ids[:,i])*is_gen
        ## Correct for the end of the last: 
        valid_inds = all_ids[is_id,i]
        valid = all_intervals[is_id]
        if shift != 0: 
            ## Pull out the last entry
            last_entry = all_intervals[lv_ind[i]][-1]-1 ## -1 because of interval indexing! 
            last_entry_id = lv_id[i]
            ## Put into the right format to be combined with other data: 
            valid_inds_prepend = np.array([last_entry_id])
            valid_prepend = np.array([[last_entry,last_entry+1]])
            ## Now we will concatenate to account for the gap between the previous trajectory's end and this one's beginning. 
            valid_inds = np.concatenate((valid_inds_prepend,valid_inds))
            valid = np.concatenate((valid_prepend,valid),axis=0)
        ## The elements of interest: intervals and relevant indices
        interval =[np.arange(inter[0],inter[-1]) for inter in valid]
        idi = np.concatenate([np.ones(len(inter))*valid_inds[intnb] for intnb,inter in enumerate(interval)])
        ## Calculate the useful coordinates of each trajectory:
        coord = np.concatenate((2*idi[:,None],2*idi[:,None]+1),axis = 1).astype(int)
        intervals.append(interval)
        ids.append(idi)
        coords.append(coord)
        ## Pull out the last valid trajectory BEFORE the start of this one. 
        
    return intervals,ids,coords 

#e Define a mapping from data+intervals+mask+signature into a set of trajectories, one for each mouse: 
def _trajectory_cost(vals,intervals,all_ids,sigma):
    if len(all_ids) == 0:
        return np.array([[np.nan,np.nan]]),[0,0]
    else:
        all_intervals,idslength,all_coords = compile_intervals(all_ids,intervals)
        all_fid = [np.nansum(abs(1-idi*(1-i))) for i,idi in enumerate(idslength)]
        ## Render each trajectory: 
        all_traj = []
        all_cost = []
        for i in range(2):
            intcont = np.concatenate(all_intervals[i])
            relevant_vals= vals[intcont[:,None],all_coords[i]]
            if relevant_vals.shape[0] == 1: 
                all_traj.append(relevant_vals)
                fidcost = intervals[len(all_ids)-1,:][-1]-intervals[0,:][0]-all_fid[i]
                all_cost.append(fidcost)
            else:
                interpf = interp1d(intcont+1e-3*i,relevant_vals,axis = 0,fill_value = 'extrapolate')
                interp = interpf(np.arange(intcont[0],intcont[-1]+1))
                all_traj.append(interp)
                tvcost = sum(np.linalg.norm(np.diff(interp,axis = 0),axis = 1))
                fidcost = intervals[len(all_ids)-1,:][-1]-intervals[0,:][0]-all_fid[i]
                cost = tvcost+sigma*fidcost
                all_cost.append(cost)
        return all_traj,all_cost

def trajectory_cost(vals,intervals,mask,signature,sigma):
    all_ids = process_signature(signature,mask)
    return _trajectory_cost(vals,intervals,all_ids,sigma)

## Return just the trajectory 
def _render_trajectory(vals,intervals,all_ids):
    all_intervals,ids,all_coords = compile_intervals(all_ids,intervals)
    all_traj = []
    all_cost = []
    for i in range(2):
        intcont = np.concatenate(all_intervals[i])
        interpf = interp1d(intcont+1e-3*i,vals[intcont[:,None],all_coords[i]],axis = 0,fill_value = 'extrapolate')
        interp = interpf(np.arange(intcont[0]+1,intcont[-1]))
        all_traj.append(interp)
    return all_traj

def render_trajectory(vals,intervals,mask,signature):
    all_ids = process_signature(signature,mask)
    return _render_trajectory(vals,intervals,all_ids)


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

def last_valid(active):
    '''
    A function to take an id set and return the indices of the last non-nan entries, and the values of those entries. 
    Parameters: 
    active (array): an array of 0,1, and nan (i.e. a float array) that codes the identity of a segment at a given time, should it exist (if not, we give it a nan)

    Returns:
    last_vald: an array of the last valid entry indices. 
    active[]: an array of the values at those indices. 
    '''
    if len(active) == 0:
        return [0,0],np.array([0,1])
    else:
        length_index = np.arange(len(active))
        ## Initialize the last time interval that you care about: 
        last_valid = [length_index[~np.isnan(active[:,0])][-1],length_index[~np.isnan(active[:,1])][-1]]
        return last_valid,active[np.array(last_valid),np.array([0,1]).astype(int)]
def first_valid(active):
    '''
    A function to take an id set and return the indices of the first non-nan entries, and the values of those entries. 
    Parameters: 
    active (array): an array of 0,1, and nan (i.e. a float array) that codes the identity of a segment at a given time, should it exist (if not, we give it a nan)

    Returns:
    first_valid: an array of the last valid entry indices. 
    active[]: an array of the values at those indices. 
    '''
    if len(active) == 0:
        return [0,0],np.array([0,1])
    else:
        length_index = np.arange(len(active))
        ## Initialize the last time interval that you care about: 
        first_valid = [length_index[~np.isnan(active[:,0])][0],length_index[~np.isnan(active[:,1])][0]]
        return first_valid,active[np.array(first_valid),np.array([0,1]).astype(int)]

def _heuristic_trajectory(trajraw,intervals,mask,all_ids,odd = 0):
    '''
    A function that takes as input the parameters you would need to generate a trajectory, generates an answer with a heuristic, and then returns the cost associated with the heuristic. 
    '''
    ## We need to handle the edge case where the signature is of odd length. 
    if odd: 
        save_val = all_ids[-1,0] ## save the value that should be preserved. 
        all_ids = all_ids[:-1,:]
    threshold = 15 
    T1 = 1 
    ## We will fill in this signature: 
    to_search = np.arange(len(all_ids),len(intervals-1*odd))
   
    ## Initialize the last time interval that you care about: 
    last_valid_ind,last_valid_ids = last_valid(all_ids)
    for i in to_search:
        ## Now for each timestep: 
        ### Look at feasibility:
        dists = [val_dist(trajraw,intervals,mask,i,m,last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
        time = [val_time(intervals,mask,i,m,last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
        dists = list(np.array(dists)/np.array(time))
        entry = np.array([float(dist< threshold) for dist in dists])
        entry[np.where(np.isnan(dists))]  = np.nan
        ## Everyhting checks out. 
        if sum(entry) == 2:
            entry_array = np.array([0,1])[None,:]
            all_ids = np.concatenate((all_ids,entry_array),axis = 0)
            last_valid_ind,last_valid_ids = last_valid(all_ids)
        ## Now we consider switches, dumps. 
        else:
            if np.nansum(entry) == 1:
                entry_array = np.array([0.,1.])[None,:]
                entry_array[:,np.where(np.array(entry)!=1)[0][0]] = np.nan
                all_ids = np.concatenate((all_ids,entry_array),axis = 0)
                last_valid_ind,last_valid_ids = last_valid(all_ids)
            ## At least one is valid, neither accepted:
            elif np.nansum(entry) == 0:
                ## First try with the pasts flipped:  
                crossdists = [val_dist(trajraw,intervals,mask,i,abs(1-m),last_valid_ind[m],last_valid_ids[m]) for m in range(2)]
                crossval = np.array(dists) > T1*np.array(crossdists)
                entry_array = np.array([1,0.])[None,:]
                if sum(crossval) == 2:
                    pass
                else:
                    entry_array[:,np.where(np.array(crossval)!=1)[0][0]] = np.nan
                all_ids = np.concatenate((all_ids,entry_array),axis = 0)
                last_valid_ind,last_valid_ids = last_valid(all_ids)
        ## Restore. 
        if odd and (i == to_search[0]): 
            all_ids[-1,0] = save_val
    return all_ids

def heuristic_trajectory(trajraw,intervals,mask,signature):
    '''
    A function that takes as input the parameters you would need to generate a trajectory, generates an answer with a heuristic, and then returns the cost associated with the heuristic. 
    '''
    all_ids = process_signature(signature,mask)
    
    return _heuristic_trajectory(trajraw,intervals,mask,all_ids,odd = len(signature)%2)           

def _heuristic_cost(vals,intervals,mask,init_ids,sigma):
    length = len(init_ids)
    if length == len(intervals): 
        return np.array([[np.nan,np.nan]]),[0,0],np.array([[np.nan,np.nan]])
    else:  
        all_ids = _heuristic_trajectory(vals,intervals,mask,init_ids)
        ## TODO:We should clip to only calculate the cost on the part of the trajectory that is not included in the signature.   
        ## TODO:Handle case with complete signature 
        all_intervals,idslength,all_coords = compile_intervals(all_ids,intervals,length)
        all_fid = [np.nansum(abs(1-idi*(1-i))) for i,idi in enumerate(idslength)]
        all_traj = []
        all_cost = []
        for i in range(2):
            intcont = np.concatenate(all_intervals[i])
            relevant_vals= vals[intcont[:,None],all_coords[i]]
            if relevant_vals.shape[0] == 1: 
                all_traj.append(relevant_vals)
                fidcost = intervals[-1,:][-1]-intervals[length,:][0]-all_fid[i]
                all_cost.append(fidcost)
            else:
                interpf = interp1d(intcont+1e-3*i,relevant_vals,axis = 0,fill_value = 'extrapolate')
                interp = interpf(np.arange(intcont[0],intcont[-1]+1))
                all_traj.append(interp)
                tvcost = sum(np.linalg.norm(np.diff(interp,axis = 0),axis = 1))
                fidcost = intervals[-1,:][-1]-intervals[length,:][0]-all_fid[i]
                cost = tvcost+sigma*fidcost
                all_cost.append(cost)
        return all_traj,all_cost,all_ids
    
    
def heuristic_cost(vals,intervals,mask,signature,sigma):
    length = int(np.ceil(len(signature)/2.))
    if length%2 == 1:
        odd = True
    else:
        odd = False
    if length == len(intervals): 
        return np.array([[np.nan,np.nan]]),[0,0]
    else:  
        all_ids = heuristic_trajectory(vals,intervals,mask,signature)
        ## TODO:We should clip to only calculate the cost on the part of the trajectory that is not included in the signature.   
        ## TODO:Handle case with complete signature 
        all_intervals,idslength,all_coords = compile_intervals(all_ids,intervals,length)
        all_fid = [np.nansum(abs(1-idi*(1-i))) for i,idi in enumerate(idslength)]
        all_traj = []
        all_cost = []
        for i in range(2):
            intcont = np.concatenate(all_intervals[i])
            relevant_vals= vals[intcont[:,None],all_coords[i]]
            if relevant_vals.shape[0] == 1: 
                all_traj.append(relevant_vals)
                fidcost = intervals[-1,:][-1]-all_intervals[i][0]-all_fid[i] ## unlike with TV, we don't want to double count. 
                all_cost.append(fidcost)
            else:
                interpf = interp1d(intcont+1e-3*i,relevant_vals,axis = 0,fill_value = 'extrapolate')
                interp = interpf(np.arange(intcont[0],intcont[-1]+1))
                all_traj.append(interp)
                tvcost = sum(np.linalg.norm(np.diff(interp,axis = 0),axis = 1))
                fidcost = intervals[-1,:][-1]-all_intervals[i][0]-all_fid[i] ## unlike with TV, we don't want to double count. 
                cost = tvcost+sigma*fidcost
                all_cost.append(cost)
        return all_traj,all_cost
    
def fake_data(length,goodpoints):
    '''
    A function to create fake data that follows the structure of what is actually going on so we can finally find these freaking bugs. 
    '''
    indices = []
    for i in range(10):
        lenindex = np.array(goodpoints)[:,None]
        ones = np.ones((len(lenindex),1))
        indices.append(np.concatenate((lenindex,ones*i),axis = 1).astype(int))
    set0 = np.ones((length,2))
    set1 = np.ones((length,2))*-1
    beh0 = np.cumsum(set0,axis = 0)
    beh1 = np.cumsum(set1,axis = 0)
    traj = np.concatenate((beh0,beh1),axis = 1)
    return traj,indices

def fake_data_switch(length,goodpoints,switchpoints):
    '''
    A function to create fake data that follows the structure of what is actually going on so we can finally find these freaking bugs. 
    '''
    indices = []
    for i in range(10):
        lenindex = np.array(goodpoints)[:,None]
        ones = np.ones((len(lenindex),1))
        indices.append(np.concatenate((lenindex,ones*i),axis = 1).astype(int))
    set0 = np.ones((length,2))
    set1 = np.ones((length,2))*-1
    beh0 = np.cumsum(set0,axis = 0)
    beh1 = np.cumsum(set1,axis = 0)
    traj = np.concatenate((beh0,beh1),axis = 1)
    switchpoints_vec = np.array([[i in switchpoints] for i in range(length)])
    logical_switchpoints_vec = -1*(switchpoints_vec*2-1)
    straj = traj*logical_switchpoints_vec
    return straj,indices


def fake_data_adv(length,goodpoints,switchpoints):
    '''
    A function to create fake data that follows the structure of what is actually going on so we can finally find these freaking bugs. 
    '''
    indices = []
    for i in range(10):
        lenindex = np.array(goodpoints)[:,None]
        ones = np.ones((len(lenindex),1))
        indices.append(np.concatenate((lenindex,ones*i),axis = 1).astype(int))
    set0 = np.random.normal(np.zeros((length,2)))*0.2
    set1 = np.random.normal(np.zeros((length,2)))*0.2
    beh0 = np.cumsum(set0,axis = 0)+1
    beh1 = np.cumsum(set1,axis = 0)-1
    traj = np.concatenate((beh0,beh1),axis = 1)
    switchpoints_vec = np.array([[i in switchpoints] for i in range(length)])
    logical_switchpoints_vec = -1*(switchpoints_vec*2-1)
    straj = traj*logical_switchpoints_vec
    return straj,indices

def signature_cost(trajraw,intervals,mask,signature,sigma):
    ## This function simply calculates the cost on the signature, and adds to it the prospective greedy cost given by the heuristic. 
    firsttraj,firstcost = trajectory_cost(trajraw,intervals,mask,signature,sigma)
    sectraj,seccost = heuristic_cost(trajraw,intervals,mask,signature,sigma)
    return np.sum(firstcost)+np.sum(seccost) 
    
if __name__ == '__main__':
    
    ## Import a preprocessed dataset so we can work quickly. 
    datapath = '../data/all_data_finetuned_votes'
    all_datasets = joblib.load(datapath) # Contains all 30 days, already preprocessed. 
    ## We will work with one to begin with:
    for dataset in [all_datasets[19]]:
        indices = dataset.allowed_index_full
        for nb_part in [0]:
            #intervals,mask= ind_to_dict_split(indices,nb_part)
        
            ## Now render the appropriate trajectory from a signature+ segment dictionary
            vraw,mraw = dataset.select_trajectory(nb_part),dataset.select_trajectory(nb_part+5)
            #trajraw = np.concatenate((vraw,mraw),axis=1)
            trajraw,indices = fake_data_adv(20,[0,2,3,5,6,7,9,10,11,12,14,15,18,19],[5,6,7,14,15])
            intervals,mask= ind_to_dict_split(indices,nb_part)
            init_sig = np.ones(2,)
            init_sig = np.array([1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1])

            ## First we calculate a heuristic trajectory: 
            hids = heuristic_trajectory(trajraw,intervals,mask,init_sig)
            signature0 = id_to_sig(hids)
            signature1 = np.array([2,2,2,1,1,1,1,1,1,1])-1
            #signature = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 0, 1,])
            #0signature = np.ones(len(signature))
            signature1 = np.array([2, 2, 0, 2, 0, 0, 2, 2, 0, 2])
            ## Now we calculate cost on it: 
            sigma = 1
            for z in [5]:
                fulltraj,fullcost = trajectory_cost(trajraw,intervals,mask,signature0,sigma)
                #comptraj,compcost = trajectory_cost(trajraw,intervals,mask,signature1,sigma)
                firsttraj,firstcost = trajectory_cost(trajraw,intervals,mask,signature0[:z],sigma)
                sectraj,seccost = heuristic_cost(trajraw,intervals,mask,signature0[:z],sigma)
                print(fullcost[0],(firstcost[0],seccost[0]))
                print(fullcost[0],(firstcost[0],seccost[0]))
            #print(fullcost,compcost)

            m = 1
            fig,ax = plt.subplots(2,1,figsize = (8,8))
            ax[0].plot(fulltraj[m][:,0],fulltraj[m][:,1])
            ax[0].plot(comptraj[m][:,0],comptraj[m][:,1])
            [ax[i].set_aspect('equal') for i in range(2)]
            plt.show() 

            ### Now initialize the signature: 
            #signature = np.ones(2,)

            #cost = signature_cost(trajraw,intervals,mask,signature,2.5)
            #print(cost)
            #signature = np.array([1,1,0,0,0,0])
            #cost = signature_cost(trajraw,intervals,mask,signature,2.5)
            #signature = np.array([1,1,1,1,1,1])
            #cost = signature_cost(trajraw,intervals,mask,signature,2.5)
