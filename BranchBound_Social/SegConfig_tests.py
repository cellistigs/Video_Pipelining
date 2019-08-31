import numpy as np
from SegConfig import *

def check_optimum(config,test):
    ## Find the groundtruth (unswitched) configuration: 
    gtquery = generate_groundtruth(test) 
    ## Also get the current base configuration: 
    basequery = config.base_config
    
    ## We will perform two checks. 1. Check that the optimum cost with optimal lagrange is higher for the ground truth than for all local perturbations. 2. Check that correcting the base configuration towards the ground truth leads to an improving cost. 

    ## 1: 
    ## First creat a dummy query configuration: 
    dummyquery = np.zeros(basequery.shape)
    dummyquery[:,:] = basequery
    dummyquery[np.where(gtquery!=-2)] = gtquery[np.where(gtquery!=-2)]
    ## Record the cost: 
    gt_cost = config.full_cost(dummyquery.astype(int))
    ## Initialize the difference array: 
    diff1 = np.zeros((dummyquery.shape[0],dummyquery.shape[1],3))
    ## Now iterate through changing one entry at a time: 
    for j in range(len(dummyquery)):
        for i in range(2):
            ## Check if this is allowed: 
            if dummyquery[j,abs(1-i)] != -1:
                ## If allowed, create an array of allowed values here: 
                allowed = np.concatenate((config.segs[j].allowed,[-1])) 
                for k,kval in enumerate(allowed): 
                    pertquery = np.zeros(dummyquery.shape)
                    pertquery[:,:] = dummyquery
                    pertquery[j,i] = kval 
                    pert_cost = config.full_cost(pertquery.astype(int))
                    diff1[j,i,k] = pert_cost-gt_cost
    ## First creat a dummy query configuration: 
    dummyquery = np.zeros(basequery.shape)
    dummyquery[:,:] = basequery
    ## Record the cost: 
    ## Find the length of the path from base to ground truth: 
    wa,wb = np.where(gtquery!=-2)
    iterinds = zip(wa,wb)
    ## Initialize the difference array: 
    diff2 = np.zeros(len(wa))
    ## Now iterate through changing one entry at a time: 
    for j,jinds in enumerate(iterinds):
        dummyquery[jinds] = gtquery[jinds] 
        pert_cost = config.full_cost(dummyquery.astype(int))
        diff2[j] = pert_cost-gt_cost
    return diff1,diff2

def linearize_path(opt,test):
    ## Get the root node: 
    node = opt.root_node
    ## Get the groundtruth query: 
    gt_query = opt.conf.return_checked(generate_groundtruth(test))
    ## Convert this to a signature: 
    bump = gt_query+1
    sig = bump[:,0]+bump[:,1]*3
    ## Now iterate through the whole segment set once: 
    all_tings = []
    for i in range(opt.conf.nb_segs-2):
        children = node.children()
        childqs = [c.signature for c in children]
        childqs_last = [q[-1] for q in childqs]
        childindex = np.where(childqs_last == sig[i])[0][0]
        node = children[childindex]
        boundchild = np.zeros((1,9))*np.nan
        for l,leaf in enumerate(childqs_last):
            val = opt.boundopt1(children[l],depth = 4)
            boundchild[0,leaf]=val
    
        boundchild = boundchild-boundchild[0,sig[i]]
        all_tings.append(boundchild)
    return np.array(all_tings)# Lay out an expansion around the ground truth at every point. 

def elementcheck_simple(opt,query=None,split = 1,duration = 5):
    ## Prep: padding is adaptive. 
    prepadding = np.max([np.min([1,(split-1)]),0])
    postpadding = np.max([np.min([1,(opt.conf.nb_segs-1-(split+duration))]),0])
    ## Get array versions of the full cost and boundopt1 cost, to compare elementwise and see what is going on. 
    ## first the full cost: 
    ## The full cost is made of the intra array, the inter array, the null array and the switch array: 
    f_intra= opt.conf.get_intra(query=query)
    f_inter = np.concatenate((opt.conf.get_inter(query=query),np.zeros((1,2))),axis = 0)
    f_null = opt.conf.full_nullcost_array(query=query)       
    f_switch = opt.conf.full_switchcost_array(query=query)
    f_intra[np.isnan(f_intra)] = 0

    ## now boundopt1 (more complicated) 
    ## We will get three different sets of arrays, corresponding to the pre-set, the during set, and the after set. 
    ## Pre set: 
    b_intra_pre = opt.conf.get_intra(query=query)[:split]
    b_inter_pre = np.concatenate((opt.conf.get_inter_cut(query=query[:split],end = split),np.zeros((prepadding,2))),axis = 0)
    b_null_pre = opt.conf.full_nullcost_array(query=query)[:split]
    b_switch_pre = opt.conf.full_switchcost_array(query=query)[:split]
    b_intra_pre[np.isnan(b_intra_pre)]=0

    ## During set: treat as a single entity because it's easier for now:  
    splitend = split+duration
    b_array_opt = opt.estopt[split:splitend]

    ## After set: 
    b_intra_post = opt.conf.get_intra(query=opt.boundquery)[splitend:]
    b_inter_post = np.concatenate((opt.conf.get_inter_cut(query = opt.boundquery[splitend:],start = splitend),np.zeros((postpadding,2))),axis = 0)
    b_null_post = opt.conf.full_nullcost_array(query=opt.boundquery)[splitend:]
    b_switch_post = opt.conf.full_switchcost_array(query=opt.boundquery)[splitend:]
    b_intra_post[np.isnan(b_intra_post)]=0

    b_bridge_post = opt.conf.bridge_cost_opt(query=opt.boundquery,split = splitend)

    b_total = np.sum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre)+np.sum(b_array_opt)+np.sum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post)+np.sum(b_bridge_post)
    ## Get corresponding node: 
    node = opt.query_to_node(split,query = query)
    #print(b_total-opt.boundopt1(node,depth = duration),opt.conf.full_cost())

    return b_total-opt.boundopt1(node,depth = duration)

## Let's see what happens when we replace optimal estimation of the middle segment with real estimation. This is a sanity check agianst the full cost computed on boundqueries. Note liberay mutation of the boundquery attribute of opt. beware. Reset after.  
def elementcheck_realistic(opt,query=None,split = 1,duration = 5):
    ######## 
    #### OPT MUTATION HAPPENING
    if query is not None:
        opt.boundquery = query
    #######
    ## Prep: padding is adaptive. 
    prepadding = np.max([np.min([1,(split-1)]),0])
    postpadding = np.max([np.min([1,(opt.conf.nb_segs-1-(split+duration))]),0])
    ## Get array versions of the full cost and boundopt1 cost, to compare elementwise and see what is going on. 
    ## first the full cost: 
    ## The full cost is made of the intra array, the inter array, the null array and the switch array: 
    f_intra= opt.conf.get_intra(query=query)
    f_inter = np.concatenate((opt.conf.get_inter(query=query),np.zeros((1,2))),axis = 0)
    f_null = opt.conf.full_nullcost_array(query=query)       
    f_switch = opt.conf.full_switchcost_array(query=query)
    f_intra[np.isnan(f_intra)] = 0

    ## now boundopt1 (more complicated) 
    ## We will get three different sets of arrays, corresponding to the pre-set, the during set, and the after set. 
    ## Pre set: 
    b_intra_pre = opt.conf.get_intra(query=query)[:split]
    b_inter_pre = np.concatenate((opt.conf.get_inter_cut(query=query[:split],end = split),np.zeros((prepadding,2))),axis = 0)
    b_null_pre = opt.conf.full_nullcost_array(query=query)[:split]
    b_switch_pre = opt.conf.full_switchcost_array(query=query)[:split]
    b_intra_pre[np.isnan(b_intra_pre)]=0

    ## During set: treat as a single entity because it's easier for now:  
    splitend = split+duration
    #b_array_opt = opt.estopt[split:splitend]
    print(splitend,'trim?')
    b_array_mid = opt.conf.cut_cost(query[split:splitend],start = split,end = splitend)
    print(np.sum(b_array_mid),'mid')


    ## After set: 
    b_intra_post = opt.conf.get_intra(query=opt.boundquery)[splitend:]
    b_inter_post = np.concatenate((opt.conf.get_inter_cut(query = opt.boundquery[splitend:],start = splitend),np.zeros((postpadding,2))),axis = 0)
    b_null_post = opt.conf.full_nullcost_array(query=opt.boundquery)[splitend:]
    b_switch_post = opt.conf.full_switchcost_array(query=opt.boundquery)[splitend:]
    b_intra_post[np.isnan(b_intra_post)]=0
    print(np.sum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post),'post')

    #b_bridge_post = opt.conf.bridge_cost(query=opt.boundquery,split = splitend)
    #b_bridge_pre = opt.conf.bridge_cost(query=opt.boundquery,split=split)
    b_bridge = opt.conf.bridge_cost_border(query=opt.boundquery,splits = [split,splitend])

    b_total = np.sum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre)+np.sum(b_array_mid)+np.sum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post)+np.sum(b_bridge)
    #print(b_bridge_pre,b_bridge_post)
    return b_total-opt.conf.full_cost(query)

## Last check: let's see if we can get the behavior on groundtruth to be accurate to that groundtruth. Now we test not on arbitrary queries but only on the groundtruth. boundquery is set to the groundtruth too. The heuristic cost lower bounds the full cost and we can account for all terms.  
def elementcheck_groundtruth(opt,test,split = 1,duration = 5):
    ## First get the groundtruth query: 
    query = opt.conf.return_checked(generate_groundtruth(test))
    ######## 
    #### OPT MUTATION HAPPENING
    opt.boundquery = opt.conf.work_config
    #workconfig = opt.conf.work_config
    #opt.conf.work_config = query
    #######
    ## Prep: padding is adaptive. 
    prepadding = np.max([np.min([1,(split-1)]),0])
    postpadding = np.max([np.min([1,(opt.conf.nb_segs-1-(split+duration))]),0])
    ## Get array versions of the full cost and boundopt1 cost, to compare elementwise and see what is going on. 
    ## first the full cost: 
    ## The full cost is made of the intra array, the inter array, the null array and the switch array: 
    f_intra= opt.conf.get_intra(query=query)
    f_inter = np.concatenate((opt.conf.get_inter(query=query),np.zeros((1,2))),axis = 0)
    f_null = opt.conf.full_nullcost_array(query=query)       
    f_switch = opt.conf.full_switchcost_array(query=query)
    f_intra[np.isnan(f_intra)] = 0

    ## now boundopt1 (more complicated) 
    ## We will get three different sets of arrays, corresponding to the pre-set, the during set, and the after set. 
    ## Pre set: 
    b_intra_pre = opt.conf.get_intra(query=query)[:split]
    b_inter_pre = np.concatenate((opt.conf.get_inter_cut(query=query[:split],end = split),np.zeros((prepadding,2))),axis = 0)
    b_null_pre = opt.conf.full_nullcost_array(query=query)[:split]
    b_switch_pre = opt.conf.full_switchcost_array(query=query)[:split]
    b_intra_pre[np.isnan(b_intra_pre)]=0
    print(np.sum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre)+np.sum(b_intra_pre),'pre)')

    ## During set: treat as a single entity because it's easier for now:  
    splitend = split+duration
    #b_array_opt = opt.estopt[split:splitend]
    b_intra_opt,b_inter_opt = opt.conf.estimate_optimal_post_array(query = opt.conf.work_config[split:splitend],start = split,end = splitend)
    print(b_intra_opt,b_inter_opt,'optarrays')
    #b_intra_opt = b_intra[split:splitend]
    #b_inter_opt = b_inter[split:splitend]

    #b_array_mid = opt.conf.cut_cost(query[split:splitend],start = split,end = splitend)
    #print(np.sum(b_array_mid),'mid')


    ## After set: 
    b_intra_post = opt.conf.get_intra(query=opt.boundquery)[splitend:]
    b_inter_post = np.concatenate((opt.conf.get_inter_cut(query = opt.boundquery[splitend:],start = splitend),np.zeros((postpadding,2))),axis = 0)
    b_null_post = opt.conf.full_nullcost_array(query=opt.boundquery)[splitend:]
    b_switch_post = opt.conf.full_switchcost_array(query=opt.boundquery)[splitend:]
    b_intra_post[np.isnan(b_intra_post)]=0

    #b_bridge_post = opt.conf.bridge_cost(query=opt.boundquery,split = splitend)
    #b_bridge_pre = opt.conf.bridge_cost(query=opt.boundquery,split=split)
    b_bridge_opt = opt.conf.bridge_cost_border_preopt(query=opt.boundquery,splits = [split,splitend])
    b_bridge = opt.conf.bridge_cost_border(query=opt.boundquery,splits = [split,splitend])

    b_total = np.sum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre)+np.sum(b_intra_opt)+np.sum(b_inter_opt)+np.sum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post)+np.sum(b_bridge_opt)
    intras = np.sum(b_intra_pre)+np.sum(b_intra_opt)+np.sum(b_intra_post)
    inters = np.sum(b_inter_pre)+np.sum(b_bridge)+np.sum(b_inter_opt)+np.sum(b_inter_post)
    intra_base = opt.conf.get_intra_cut(query=query[split:splitend],start = split,end=splitend)
    intra_base[np.isnan(intra_base)] = 0
    node = opt.query_to_node(split,query = query)
    print(opt.conf.return_checked(opt.s2q(node.signature))[split:splitend],'config?')
    #print(np.sum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre),'pre')
    print(np.sum(b_intra_opt)+np.sum(b_inter_opt),'opt')
    print(split,splitend,'split')
    
    return b_total-opt.conf.full_cost(query),b_total-opt.conf.full_cost(),b_total-opt.boundopt1(node,depth = duration)


def elementcheck(opt,query=None,split = 1,duration = 5):
    ## Prep: padding is adaptive. 
    prepadding = np.max([np.min([1,(split-1)]),0])
    postpadding = np.max([np.min([1,(opt.conf.nb_segs-1-(split+duration))]),0])
    ## Get array versions of the full cost and boundopt1 cost, to compare elementwise and see what is going on. 
    ## first the full cost: 
    ## The full cost is made of the intra array, the inter array, the null array and the switch array: 
    f_intra= opt.conf.get_intra(query=query)
    f_inter = np.concatenate((opt.conf.get_inter(query=query),np.zeros((1,2))),axis = 0)
    f_null = opt.conf.full_nullcost_array(query=query)       
    f_switch = opt.conf.full_switchcost_array(query=query)
    #print(np.nansum(f_intra),np.sum(f_inter),np.sum(f_null),np.sum(f_switch))
    print(np.nansum(f_intra)+np.sum(f_inter)+np.sum(f_null)+np.sum(f_switch),opt.conf.full_cost(query= query),'full')
    assert np.nansum(f_intra)+np.sum(f_inter)+np.sum(f_null)+np.sum(f_switch) == opt.conf.full_cost(query= query)
    f_intra[np.isnan(f_intra)] = 0
    f_array = f_intra+f_inter+f_null 

    ## now boundopt1 (more complicated) 
    ## We will get three different sets of arrays, corresponding to the pre-set, the during set, and the after set. 
    ## Pre set: 
    b_intra_pre = opt.conf.get_intra(query=query)[:split]
    b_inter_pre = np.concatenate((opt.conf.get_inter_cut(query=query[:split],end = split),np.zeros((prepadding,2))),axis = 0)
    b_null_pre = opt.conf.full_nullcost_array(query=query)[:split]
    b_switch_pre = opt.conf.full_switchcost_array(query=query)[:split]
    print(np.nansum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre),opt.conf.cut_cost(query=query[:split],end = split),'pre')
    #assert np.nansum(b_intra_pre)+np.sum(b_inter_pre)+np.sum(b_null_pre)+np.sum(b_switch_pre) == opt.conf.cut_cost(query=query[:split],end = split)
    b_intra_pre[np.isnan(b_intra_pre)]=0
    b_array_pre= b_intra_pre+b_inter_pre+b_null_pre+b_switch_pre

    ## During set: treat as a single entity because it's easier for now:  
    splitend = split+duration
    b_array_opt = opt.estopt[split:splitend]

    ## After set: 
    b_intra_post = opt.conf.get_intra(query=opt.boundquery)[splitend:]
    b_inter_post = np.concatenate((opt.conf.get_inter_cut(query = opt.boundquery[splitend:],start = splitend),np.zeros((postpadding,2))),axis = 0)
    b_null_post = opt.conf.full_nullcost_array(query=opt.boundquery)[splitend:]
    b_switch_post = opt.conf.full_switchcost_array(query=opt.boundquery)[splitend:]
    #print(np.nansum(b_intra_post),np.sum(b_inter_post),np.sum(b_null_post),np.sum(b_switch_post),opt.conf.cut_cost(query=query[splitend:],start = splitend))
    #print(np.nansum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post),opt.conf.cut_cost(query=query[splitend:],start = splitend),'post')
    #assert np.nansum(b_intra_post)+np.sum(b_inter_post)+np.sum(b_null_post)+np.sum(b_switch_post) == opt.conf.cut_cost(query=query[splitend:],start = splitend)
    b_intra_post[np.isnan(b_intra_post)]=0
    print(np.sum(b_intra_post),np.sum(b_inter_post),np.sum(b_null_post),np.sum(b_switch_post),'postcomps')
    print(b_intra_post.shape,b_inter_post.shape,b_null_post.shape,b_switch_post.shape)
    b_array_post = b_intra_post+b_inter_post+b_null_post+b_switch_post 
    print(np.sum(b_array_post),'here')

    b_bridge_post = opt.conf.bridge_cost_opt(query=opt.boundquery,split = splitend)
    b_total = np.sum(b_array_pre)+np.sum(b_array_opt)+np.sum(b_array_post)+b_bridge_post
    ## Get corresponding node: 
    node = opt.query_to_node(split,query = query)
    #print(b_total-opt.boundopt1(node,depth = duration),opt.conf.full_cost())
    
    return b_total-opt.boundopt1(node,depth = duration)

## Node Checks: 
## The comprehensive check of sanity of heuristic behavior. No guarantees of consistency with true cost, but guaranteed internal consistency. 
def path_check_elementwise(opt):
    duration = 5
    node = opt.root_node
    ## Trawl signatures randomly. 
    for i in range(node.length):
        print('checking at length '+str(i))
        children = node.children()
        nb_children = len(children)
        cind = np.random.choice(np.arange(nb_children))
        node = children[cind]
        signature = node.signature
        query = opt.s2q_default(signature,opt.conf.work_config)
        ## Got a query. 
        vals = [elementcheck_simple(opt,query = query,split =j,duration = d) for j in range(node.length-duration) for d in range(duration)[1:]]
        assert np.all(np.array(vals)<1e-5)
    
def path_check_optcost(opt,start=None,end = None):
    duration = 5
    node = opt.root_node
    ## Trawl signatures randomly. 
    vals = []
    for i in range(node.length):
        print('checking at length '+str(i))
        children = node.children()
        nb_children = len(children)
        cind = np.random.choice(np.arange(nb_children))
        node = children[cind]
        signature = node.signature
        query = opt.s2q_default(signature,opt.conf.work_config)
        print(query[start:end])
        query = np.ones((5,2))*-1
        ## Got a query. 
        val = opt.conf.estimate_optimal_post(query=query[start:end],start = start,end = end)
        vals.append(val)
    return vals
## Check that a path explored through bound operations generates costs in keeping with what we expect. 
def path_check_elementwise_real(opt):
    duration = 5
    node = opt.root_node
    ## Trawl signatures randomly. 
    for i in range(node.length):
        print('checking at length '+str(i))
        children = node.children()
        nb_children = len(children)
        cind = np.random.choice(np.arange(nb_children))
        node = children[cind]
        signature = node.signature
        query = opt.s2q_default(signature,opt.conf.work_config)
        ## Got a query. 
        vals = [elementcheck_realistic(opt,query = query,split =j,duration = d) for j in range(node.length-duration) for d in range(duration)]
        assert np.all(np.array(vals)<1e-5)

def path_check(opt):
    print(opt.root_node.length)
    node = opt.root_node
    ref_old = 0
    for i in range(node.length):
        children = node.children()
        nb_children = len(children)
        cind = np.random.choice(np.arange(nb_children))
        node = children[cind]
        boundcost = opt.boundopt1(node)
        signature = node.signature
        query = opt.s2q_default(signature,opt.conf.work_config)
        refcost = opt.conf.cut_cost(query[:node.depth],end = node.depth,check = False)
        cost = node.cost
        print(cost,refcost)
        ref_old = refcost
    print(opt.conf.full_cost(query))
    refcost = opt.conf.cut_cost(query)
    print(refcost)

## Simulate the optimization environment. Every time that the bounding cost is updated, make sure that the ground truth cost at all splits is feasible. 
def optimization_sim(opt,test):
    ## Initialize: 
    active_nodes = []
    active_nodes.append(opt.root_node)
    ## Recover the groundtruth query: 
    gt = opt.conf.return_checked(generate_groundtruth(test))
    ## We want a way to log solution characteristics: 
    soldict = {}
    ## Now we explore our options:
    solcounter = 0
    while len(active_nodes) > 0:
        node = active_nodes.pop()
        ## Branch: 
        candidates = opt.branch(node)
        ## Handle the case where we have a solution:
        if candidates is None:
            query = opt.s2q(node.signature)
            cost = opt.conf.full_cost(query,check = False)
            ## Decide if we will accept this solution: 
            if cost <= opt.boundcost:
                opt.boundcost = cost
                opt.boundquery = query
                ## Simulate the path along the ground truth trajectory to make sure it's feasible:
                nodes = [opt.query_to_node(s,query = gt) for s in range(opt.conf.nb_segs)]
                boundcosts_gt = [opt.boundopt1(node) for node in nodes]
                soldict[solcounter] = [opt.boundcost-boundcosts_gt,opt.conf.full_cost(gt),opt.conf.full_cost(query)] 
                solcounter +=1 
            else:
                pass
        else:
            for n,cnode in enumerate(candidates):
                l = len(candidates[0].signature)
                print(l)
                nodebound = opt.boundopt1(cnode)
                if nodebound > opt.boundcost: 
                    pass
                else:
                    active_nodes.append(cnode)
    return soldict

## Code to explicitly get the components of the cost heuristic for comparison: 
def explcost(opt,test,i):
    gt = opt.conf.return_checked(generate_groundtruth(test))
    pre = opt.conf.cut_cost(query=gt[:i],end = i) 
    post = opt.conf.cut_cost(query = opt.boundquery[i+5:],start = i+5)
    
    return [pre,post]

def optimization_diagnosis(opt,test):
    active_nodes = []
    active_nodes.append(opt.root_node)
    ## Recover the groundtruth query: 
    gt = opt.conf.return_checked(generate_groundtruth(test))
    ## Now consider zooming in on why it is that we have costs that are too high:
    nodes = [opt.query_to_node(s,query = gt) for s in range(opt.conf.nb_segs+1)]
    boundcosts_gt = [opt.boundopt1_debug(node)[0] for node in nodes]
    explcosts_gt = [np.round(opt.boundopt1_debug(node)[1],5) for node in nodes]
    explcosts = [explcost(opt,test,i) for i in range(opt.conf.nb_segs+1)]
    return explcosts_gt,explcosts,boundcosts_gt 

## Helper function to address queries that we want. Take an existing id set, and blackout arbitrary blocks from it.  
def get_blackout(query,removed): 
    ## Removed should be a list of indices. 
    assert type(removed) is list or type(removed) is np.ndarray
    if type(removed) is list:
        removed = np.array(removed)

    ## First initialize a dummy query: 
    copy = np.copy(query)
    ## Black it out: 
    copy[removed,:] = -1 
    return copy
        

    




    















