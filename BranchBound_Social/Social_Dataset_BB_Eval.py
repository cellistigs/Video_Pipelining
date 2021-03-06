import numpy as np
import joblib 
from Social_Dataset_Class_v2 import social_dataset
from Social_Dataset_BB_2 import *
from Branch_Bound.ops import *
from Branch_Bound.dataformat import *
from heapq import heappush,heappop

## Write a bound function that works with the data: 
def bound_path_real(Node_path,data,k,sigma):
    '''
    Bound operation that uses the real data. The data parameter kere can be unpacked into three subparameters: the raw trajectories, the intervals, and the masks that are generated by functions from Social_Dataset_BB_2. 
    '''
    ## Problem parameters from the data:
    trajraw = data[0]
    intervals = data[1]
    mask = data[2]
    N = np.shape(intervals)[0]*2
    d = 3 ## This is dictated by the problem. 
    base_sig = Node_path.signature
    current_depth = len(base_sig)
    search_depth = current_depth +k 
    excess_depth = search_depth-N
    k_eff = k-np.max([0,excess_depth])
    ### Check if we will be greedy/if we are at the bottom of the tree
    if k_eff == 0:
        eff_sigs = [base_sig]
    ### Otherwise we explore paths to depth k:
    else:
        recurse_sigs = np.array(np.meshgrid(*[np.arange(d) for ki in range(k_eff)])).T.reshape(-1,k_eff)
    ### Calculate cost of all elements in recursion
        eff_sigs = [base_sig+list(recurse_sig) for recurse_sig in recurse_sigs]

    all_cost = [signature_cost(trajraw,intervals,mask,np.array(sig)-1,sigma) for sig in eff_sigs]
    #print(all_cost,[np.array(sig)-1 for sig in eff_sigs],'s')
    min_cost = np.min(all_cost)
    ### Finally, we need a flag to indicate if this represents a real solution
    if excess_depth == k:
        solution = 1
    else:
        solution = 0
    return min_cost,solution

## 


## With LIFO stack (upper bound by negative inf to start): 
def BB_LIFO(data,k,sigma):
    ## First recover problem parameters from the data:
    transitions_nb = np.shape(data[1])[0]
    states_nb = 3 
    N = transitions_nb*2
    d = states_nb
    ## Now come up with a heuristic solution: 
    mincost = signature_cost(data[0],data[1],data[2],np.ones((transitions_nb)*2,),sigma)
    ## Initialize the root node:
    root = Node_path([2,2],d,N)
    ## Initialize problem parameters: 
    B = np.inf #mincost # Upper bound on solutions
    ## Now we will search the tree using BB_k: 
    node_list = [root]
    node_solution = []
    while len(node_list):
        ## Get the active node 
        node_active = node_list.pop()
        ## Branch on that node
        nodes_eval = branch_path(node_active)
        ## Evaluate each child node
        bounds = [bound_path_real(node_eval,data,k,sigma) for node_eval in nodes_eval]
        print(bounds)
        ## Evaluate if a solution was reached, and add those child nodes that satisfy the bound. 
        for child_nb,bound in enumerate(bounds):
            if bound[0] <= B+1:
                node_list.append(nodes_eval[child_nb])
                ## If the solution is as good as a previously found one, add it, if it is better replace the other. 
                if bound[1] == 1:
                    if B == bound[0]:
                        node_solution.append((nodes_eval[child_nb],bound))
                    else:
                        B = bound[0]
                        node_solution = [(nodes_eval[child_nb],bound)]
    return node_solution

## With priority queue (upper bound by negative inf to start): 
def BB_Q(data,k):
    ## First recover problem parameters from the data:
    
    transitions_nb = np.shape(data[1])[0]
    states_nb = 3 
    N = transitions_nb*2
    d = states_nb
    ## Now come up with a heuristic solution: 
    mincost = signature_cost(data[0],data[1],data[2],np.ones(transitions_nb*2,),2.5)
    ## Initialize the root node:
    root = Node_path([1,1],d,N)
    ## Initialize problem parameters: 
    B = np.inf # mincost # Upper bound on solutions
    ## Now we will search the tree using BB_k: 
    node_list = []
    heappush(node_list,(B,0,root)) 
    node_solutions = []
    entry_count = 0
    while len(node_list):
        ## Get the active node 
        parent_bound,parent_sol,node_active = heappop(node_list)
        # In this case reaching a solution should be good enough (?)
        if parent_sol == 1:
            B = parent_bound
            node_solution = (node_active,[parent_bound,parent_sol])
            return node_solution

        ## Branch on that node
        nodes_eval = branch_path(node_active)
        ## Evaluate each child node
        bounds = [bound_path_real(node_eval,data,k) for node_eval in nodes_eval]
        ## Evaluate if a solution was reached, and add those child nodes that satisfy the bound. 
        for child_nb,bound in enumerate(bounds):
            entry_count+=1
            if bound[1] == 1:
                solution_penalty = d**N 
            else:
                solution_penalty = 0 
            heappush(node_list,(bound[0],entry_count+solution_penalty,nodes_eval[child_nb]))





if __name__ == '__main__':
    
    ## Import a preprocessed dataset so we can work quickly. 
    datapath = 'data/all_data_finetuned_votes'
    all_datasets = joblib.load(datapath) # Contains all 30 days, already preprocessed. 
    ## We will work with one to begin with:
    for dataset in [all_datasets[0]]:
        indices = dataset.allowed_index_full
        indices = [ind[:1000,:] for ind in indices]
        for nb_part in [0]:
            intervals,mask= ind_to_dict_split(indices,nb_part)
            #print(intervals)
        
            ## Now render the appropriate trajectory from a signature+ segment dictionary
            vraw,mraw = dataset.select_trajectory(nb_part),dataset.select_trajectory(nb_part+5)
            trajraw = np.concatenate((vraw,mraw),axis=1)
            trajraw = trajraw[:intervals[-1][-1],:]

            #trajraw,indices = fake_data_adv(15,[0,2,3,5,6,7,9,10,11,12,14],[5,6,7])
            #intervals,mask= ind_to_dict_split(indices,nb_part)
            ## Now initialize the signature: 
            #signature = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 0, 1,])
            signature = np.ones(2,)
            instance_data = [trajraw,intervals,mask]
            sigma = 1 
            out = BB_LIFO(instance_data,2,sigma)
            signature = np.array(out[0][0].signature)
            cost = signature_cost(trajraw,intervals,mask,signature,sigma)
            gtsignature = np.ones(np.shape(out[0][0].signature))
            gtcost = signature_cost(trajraw,intervals,mask,signature,sigma)
            print(cost,gtcost)
     
            #signature = np.array([1,1,0,0,0,0])
            #cost = signature_cost(trajraw,intervals,mask,signature,sigma)
            #print(cost)
            #signature = np.array([1,1,1,1,1,1])
            #cost = signature_cost(trajraw,intervals,mask,signature,sigma)
            #print(cost)
