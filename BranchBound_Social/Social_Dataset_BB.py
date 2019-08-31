## Temporary solution
import sys
sys.path.append('../')
from Social_Dataset_Class_v2 import social_dataset,find_segments
from scipy.interpolate import interp1d
import numpy as np
import joblib

## Make a segment dictionary from an index set. Return additionally, the heuristic dataset that we should initialize to. 
def ind_to_dict(indices,nb_part):
    v_partinds,m_partinds = indices[nb_part],indices[nb_part+5]
    vsegs,msegs = find_segments(v_partinds[:,0]),find_segments(m_partinds[:,0])
    ## We should treat the sort identified above as the heuristic solution we will use to initialize our BB algorithm. 
    vsegs_full,msegs_full = [seg + [0] for seg in vsegs],[seg + [1] for seg in msegs]
    full_list = sorted(vsegs_full+msegs_full)
    ## Construct a dictionary that gives segment coordinates given an index set. 
    segdict  = {i:segment for i,segment in enumerate(full_list)}
    
    ## Give the ids for the heuristic solutions found by the original algorithm
    vsegs_id_heuristic = [np.where([full_list[j] == vsegs_full[i] for j in range(len(full_list))])[0][0] for i in range(len(vsegs_full))]

    msegs_id_heuristic = [np.where([full_list[j] == msegs_full[i] for j in range(len(full_list))])[0][0] for i in range(len(msegs_full))]
    ## Check that original ids give expected answers:
    vsegs_reconstructed = [segdict[i] for i in vsegs_id_heuristic]
    msegs_reconstructed = [segdict[i] for i in msegs_id_heuristic]
    return segdict,vsegs_reconstructed,msegs_reconstructed

## Make a segment dictionary that splits on the starts and ends of either trajectories.  
## Returns two arrays: The first indicates the breakpoints between segments. The second indicates which have of the two have data. 

## Takes two sets of segments and returns two lists: one that gives all breakpoints, and another that tells which animals have data between the breakpoints. 
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
#
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
    ## Construct a dictionary that gives segment coordinates given an index set. 
    segdict  = {i:segment for i,segment in enumerate(intervals)}
    
    ## Give the ids for the heuristic solutions found by the original algorithm
    vsegs_id_heuristic = [np.where([full_list[j] == vsegs_full[i] for j in range(len(full_list))])[0][0] for i in range(len(vsegs_full))]

    msegs_id_heuristic = [np.where([full_list[j] == msegs_full[i] for j in range(len(full_list))])[0][0] for i in range(len(msegs_full))]
    ## Check that original ids give expected answers:
    vsegs_reconstructed = [segdict[i] for i in vsegs_id_heuristic]
    msegs_reconstructed = [segdict[i] for i in msegs_id_heuristic]

## Trajectories should be rendered independently of the object: dont want to mutate existing rendering.
def render_trajectory(vals,signatures,segdict):
    ## Calculate Trajectories
    vsig,msig = signatures
    traj = []
    for mouse,sig in enumerate([vsig,msig]):
        if len(sig):
            ## Get the relevant indices 
            indices = np.concatenate([np.arange(*segdict[i][:2]) for i in sig])
            ## Get the relevant identities for each index. 
            ids = np.concatenate([np.ones(segdict[ind][1]-segdict[ind][0])*segdict[ind][2]%4 for ind in sig])
            ## Make these into coordinates we can use: 
            coords = np.concatenate((ids[:,None],ids[:,None]+1),axis = 1).astype(int)
            ## Create trajectories with interpolating functions. 
            ilen = indices[-1]+1-indices[0]
            interpf = interp1d(indices+1e-3,vals[indices[:,None],coords],axis = 0,fill_value = 'extrapolate')
            interp = interpf(np.arange(indices[0]+1,indices[-1]))
            traj.append(interp)
        else:
            traj.append([])

    return traj 

def calculate_TV_cost(vals,signatures,segdict,sigma):
    ## Calculate Trajectories
    vsig,msig = signatures
    tv = []
    fid = []
    for mouse,sig in enumerate([vsig,msig]):
        if len(sig):
            ## Get the relevant indices 
            indices = np.concatenate([np.arange(*segdict[i][:2]) for i in sig])
            if len(indices) < 2:
                tv.append(0)
                fid.append(0)
            else:
                ## Get the relevant identities for each index. 
                ids = np.concatenate([np.ones(segdict[ind][1]-segdict[ind][0])*segdict[ind][2]%4 for ind in sig])
                ## Make these into coordinates we can use: 
                coords = np.concatenate((ids[:,None],ids[:,None]+1),axis = 1).astype(int)
                ## Create trajectories with interpolating functions. 
                ilen = indices[-1]+1-indices[0]
                interpf = interp1d(indices+1e-3,vals[indices[:,None],coords],axis = 0,fill_value = 'extrapolate')
                interp = interpf(np.arange(indices[0]+1,indices[-1]))
                ## Calculate Cost:
                ### Calculate Fidelity:
                pres = len(np.where(ids == mouse))
                lfid = ilen-pres
                ### Calculate TV: 
                ltv = sum(np.linalg.norm(np.diff(interp,axis = 0),axis = 1))
                tv.append(ltv)
                fid.append(lfid)
        else:
            tv.append(0)
            fid.append(0)
            
    return sum(tv) + sigma*sum(fid) 

def sig_to_cost(data,segdict,sig,start = 0):
    sigma = 10
    ## Unpack the signature:
    full_sig = [[],[],[]]
    [full_sig[sid].append(s+start) for s,sid in enumerate(sig)]
    mousesigs = full_sig[1:]
    newcost = calculate_TV_cost(data,mousesigs,segdict,sigma)
    ## Check how much information you have thrown away: 
    chucked = full_sig[0]
    chuckcost = 10*sigma*sum([segdict[i][1]-segdict[i][0] for i in range(len(chucked))])
    return newcost+chuckcost

## If the trajectory starts in a convincing position relative to the last end for this animal, paste in the _whole_ trajectory
def test_continuity(ind0,ind1,segdict,trajraw):
    _,end,mid0 = segdict[ind0]
    start,_,mid1 = segdict[ind1]
    print(ind0,ind1,start,end)

    difference = trajraw[start,2*mid0:2*mid0+2]-trajraw[end-1,2*mid1:2*mid1+2] # here end is used in slice indexing, so we subtract one
    return np.linalg.norm(difference)




def trim_trajectory(inds,segdict):
    '''
    This function gives the effective start and end along which two trajectories can be compared.
    '''
    ind0,ind1 = inds
    start0,end0 = segdict[ind0][:2]
    start1,end1 = segdict[ind1][:2]

    return max(start0,start1),min(end0,end1)

def calculate_gap(ind0,ind1,segdict,trajraw):
    '''
    This function calculates the distance between the end of one segment [indicated by ind0] and the start of the next 
    '''

def classify_ps_interval_func(trajraw,segdict,pindex,params):
    threshbound = 100+np.exp(params[0])
    T0 = params[1]
    T1 = params[2]
    T2 = params[3]
    T3 = params[4]

    ## We will iterate through all the segments available to us: 
    nb_segments = len(segdict)
    
    curr_v,curr_m = 0,0
    curr_vals = [curr_v,curr_m]
    for nb_seg in range(nb_segments):
        entry = segdict[nb_seg]
        ## Now see if currently active segments merit comparison:
        minind,maxind = trim_trajectory(curr_vals,segdict)
        if minind >= maxind:
            pass
        else: 

            ## Finally, update current segment assignments
            ## TODO: update to reflect case where we have to switch indices. 
            pass
        curr_vals[entry[-1]] = nb_seg










if __name__ == '__main__':
    ## Import a preprocessed dataset so we can work quickly. 
    datapath = '../data/all_data_finetuned_votes'
    all_datasets = joblib.load(datapath) # Contains all 30 days, already preprocessed. 
    ## We will work with one to begin with:
    for dataset in [all_datasets[0]]:
        indices = dataset.allowed_index_full
        for nb_part in [0]:
            segdict,vsegs_reconstructed,msegs_reconstructed = ind_to_dict(indices,nb_part)
            ## Now render the appropriate trajectory from a signature+ segment dictionary
            vraw,mraw = dataset.select_trajectory(nb_part),dataset.select_trajectory(nb_part+5)
            trajraw = np.concatenate((vraw,mraw),axis=1)
            sigs = ([0,1,2,3,4,5,8],[]) 
            #vinds,minds = render_trajectory(trajraw,sigs,segdict)
            #newcost = calculate_TV_cost(trajraw,sigs,segdict,2.5)
            #print(len(segdict))
            classify_ps_interval_func(trajraw,segdict,nb_part,[0,0,0,0,0])

        print(segdict[len(segdict)-1])


