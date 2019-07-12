import sys 
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np

from Social_Dataset_utils import order_segsets,val_dist,val_time,intra_dist,intra_time,intra_tv
### Module to test the new trajectory analysis functions for branch and bound implementation. This is for two purposes. 1. To test the distance/time/rate calculation functions we have to make sure that they make sense in our segment based formulation. Test both the inter and intra segment distances. Figure out a good way to design these beforehands. Make a test object that switches segments in a way that follows statistics, likewise drops segments in a way that follows statistics. 2. To test the validity of the cost functions that we come up with. We can do this by having error rates in the segments (governed by breakpoints, switchpoints) that match the statistics of our data, and making sure they function as expected. 

## Helper function to convert from missing indices to present indices. 
def break_to_fill(breakpoints,length):
    ## Assume we will be taking in a list. 
    assert type(breakpoints) == list
    if len(breakpoints) > 0:
        assert isinstance(breakpoints[0],(int,np.integer))
        assert np.all(np.diff(breakpoints)>0); 'Must be given in ascending order'
        assert breakpoints[-1] < length; 'Breakpoint not valid for given length'
    breakpoints_ap = [-1]+breakpoints+[length] ## +1 when actually finding segments. 
    segs = [np.arange(breakpoints_ap[i]+1,breakpoints_ap[i+1]) for i in range(len(breakpoints)+1)]
    array = np.concatenate(segs)
    return array

## Helper function to generate circle coordinates for a given time point.  
## Returns a function that will take in integer indices (representing units of time) and output positions of the corresponding circles.
## This is useful for generating our groundtruth data, and checking responses agianst it. 
def circlefunc(center,period,radius):
    ## First generate the index: 
    ## Scale by the period of the rotation you would like: 
    circlecoords = lambda x: np.array([np.sin(2*np.pi*(x/float(period))),np.cos(2*np.pi*(x/float(period)))])*radius+center
    return circlecoords

## Helper function to generate raw circles
def circle(center,period,radius,length):
    ## First generate the index: 
    index = np.arange(length)
    #generate a circle function: 
    cfunc = circlefunc(center,period,radius)

    circ = list(map(cfunc,index))
    return circ
  

class Circle(object):
    '''
    An object class that instantiates instances holding fake data, consisting of two circles that are separated by some centroid distance. This will be used to test the correctness of the  
    Parameters:
    length (int): a positive integer describing how long the trajectory should be. 
    breakpoints (list): a list of two lists, describing the breakpoints in each trajectory. The number gives the index to be dropped.  
    switchpoints (list): a list of two lists, describing points where the two trajectories should switch. Should be disjoint from breakpoints. Follows "pull" logic- describe where you want *this* trajectory to take on the other trajectory's points. 
    '''
    ## This object holds the data associated with a fake dataset consisting of two spirals that have been segmented and switched. 
    def __init__(self,length,breakpoints,switchpoints):
        self.length = length
        self.breakpoints = breakpoints
        self.switchpoints = switchpoints
        self.centers = [np.array([10,10]),np.array([-10,-10])]
        self.period = 100
        self.radius = 10

        ### From this, recover representations of the data that are easier to work with: indices where you actually do have data. The easiest way to do this is probably to mock the real data representation exactly: create an "allowed_index_full" variable, as well as a "trajraw" variable, holding the relevant information. 
        self.make_index()

        # Render circle
        self.render_circle()
    
    def make_index(self):
        ## We need to make integer arrays: 
        effective_length = [self.length-len(self.breakpoints[i]) for i in range(2)] ## Eventually if we care about more than one body part on each animal we can have 10 different breakpoint sets. 
        ## Convert to filled representation: 
        filled_ids = [np.ones(effective_length[index>4])*index for index in range(10)] 
        ## We need to make corresponding index arrays of filled in segments.
        filled_ind = [break_to_fill(breakpoint,self.length) for breakpoint in self.breakpoints]
        # Now construct the full index representation:
        index_rep = [np.concatenate((filled_ind[f_ind>4][:,None],f_id[:,None]),axis = 1) for f_ind,f_id in enumerate(filled_ids)]
        self.allowed_index_full = index_rep

    def render_circle(self):
        ### Additionally construct a raw trajectory representation. 
        raw_circs = np.concatenate([circle(center,self.period,self.radius,self.length) for center in self.centers],axis = 1)
        ### We need to introduce switches. Construct a binary array that indicates when we should have switches. 
        switchpoints_binary = [np.ones((self.length,1))*i for i in range(2)]
        ### Make it live up to the name: 
        for i in range(2):
            switchpoints_binary[i][self.switchpoints[i]] =abs(1-i) 
        switchpoints_index = [np.concatenate((2*switchpoint_binary,2*switchpoint_binary+1),axis = 1) for i,switchpoint_binary in enumerate(switchpoints_binary)]
        switch_fullindex = np.concatenate(switchpoints_index,axis = 1).astype(int)
        ## Employ these indices.
        self.trajraw = raw_circs 
        self.trajswitch = self.trajraw[np.arange(self.length)[:,None],switch_fullindex] 
        ## Now put nans where the breakpoints are (for visuals)
        bindex = [np.stack((self.breakpoints[i],np.ones(len(self.breakpoints[i]))*i),axis = 1) for i in range(2)]
        all_bindex = np.concatenate(bindex,axis = 0).astype(int)
        self.traj = np.zeros(np.shape(self.trajswitch))
        self.traj[:,:] = self.trajswitch[:,:]
        self.traj[all_bindex[:,0],2*all_bindex[:,1]] = np.nan
        self.traj[all_bindex[:,0],1+2*all_bindex[:,1]] = np.nan

    def change_circparams(self,period=None,radius=None,centers=None):
        if period is not None:
            assert period >0; 'Period must be a positive float'
            self.period = period
        if radius is not None:
            assert radius >0; 'Radius must be a positive float'
            self.radius = radius
        if centers is not None:
            for center in centers:
                assert len(centers) == 2; 'Centers must be in two dimensions'
            self.centers = centers
        self.render_circle()

    def return_raw(self):
        return self.trajraw

    def return_switched(self):
        return self.trajswitch

    def return_full(self):
        return self.traj

## Define a new object class that eats a Circle Object and parameters of switch statistics for intervals. Also generates ground truth and switched distances.  
## Assume that there is a distribution of continuous detection time and a distribution of omission times (independent). This generates segments. Furthermore, assume that switches happen by flipping a bernoulli with probability p.  

class SwitchCircle(object):
       # seg[-1]-=1
    def __init__(self,length,lambseg,lambbreak,bernparam,period = None,radius = None,centers = None):
        self.length = length
        self.lambseg = lambseg
        self.lambbreak = lambbreak
        self.bern = bernparam
        self.period = period
        self.radius = radius
        self.centers = centers
        ## Take these parameters and generate intervals,ids,flips from them randomly: 
        self.random_init()
        ## Now, take intervals, ids, flips, and generate breakpoints,switchpoints that we can feed to Circle. 
        self.format_trajectory()

    ## High level formatting functions. 
    def random_init(self):
        ## First generate segmentation (initialise self.intervals_trimmed,self.ids_trimmed)
        self.generate_segmentation()
        ## Next generate switches (update self.ids_trimmed, create self.flips)
        self.generate_switches()

    ## Format the currently given intervals, ids, flips and generate trajectories from them. 
    def format_trajectory(self):
        self.format_perturbations()
        #assert self.length == self.intervals_trimmed[-1,-1]; 'interval signature must match given length'
        ## Now render the trajectory: 
        self.Circle = Circle(self.length,self.breakpoints,self.switchpoints) 
        self.Circle.change_circparams(period = self.period,radius = self.radius,centers = self.centers)
        self.trajectory = self.Circle.return_full()
        ## Get the valid trajectories attribute. 
        self.construct_valids()

    ## Slightly lower level formatting functions. 
    def generate_segmentation(self):
        ## Now we generate breakpoints for both trajectories. Don't worry about how many to generate for now.  
        segs = np.ceil(np.random.exponential(self.lambseg,size = self.length*2))
        breaks = np.ceil(np.random.exponential(self.lambbreak,size = self.length*2))
        inds = np.stack((segs,breaks),axis = 1).reshape(self.length*4)
        segpoints = (np.cumsum(inds[:self.length*2]),np.cumsum(inds[self.length*2:]))
        ## Truncate to less than the length of the trajectory: 
        segpoints_trunc = [seg[np.where(seg<self.length)] for seg in segpoints]
        ## Convert to full representation: 
        ## First add 0 and length for completeness:
        segpoints_final = [np.concatenate(([0],segs,[self.length])) for segs in segpoints_trunc]
        intervals = [np.array([[segf[i],segf[i+1]] for i in range(len(segf)-1)]) for segf in segpoints_final] 
        ## Collect segments and breaks: 
        self.presegments = [interval[0::2,:] for interval in intervals]
        breaks = [interval[1::2,:] for interval in intervals]
        intervals_pretrimmed,ids_pretrimmed = order_segsets(*self.presegments)
        self.intervals_trimmed,self.ids_trimmed = np.stack(intervals_pretrimmed),np.stack(ids_pretrimmed).astype(int)

    def generate_switches(self):    
        ## We now have a set of indices corresponding to the included and excuded segment of this dataset. 
        ## things can get confusing here. after this is applied, ids_trimmed will identify which segments actually have data, after being flipped. Flips will identify which segments *were* pasted to the other identity in order to arrive at the current data. (as opposed to, which segments *are*  the pasted ones.)
        ## Flip a bent correlated coin to determine the identity of each segment. 
        ## Get correlated probabilities
        all_segs_length = len(self.intervals_trimmed)
        priors = np.random.beta(self.bern,1-self.bern,all_segs_length)
        ## Get actual flips: 
        self.flips = np.random.binomial(1,priors,(2,all_segs_length)).T
        ## We have to recover the removal and flip indices. 
        ## First flip the relevant indices in the removal index: 
        ## Only flip those that have data to contribute.
        self.flips[np.where(np.fliplr(self.ids_trimmed==0))] = 1

        timecoord, idcoord = np.where((1-self.flips))
        flipcoord = abs(1-idcoord)
        self.ids_flipped = np.zeros(self.ids_trimmed.shape) 
        self.ids_flipped[:,:] = self.ids_trimmed[:,:]
        self.ids_trimmed[timecoord,idcoord] = self.ids_trimmed[timecoord,flipcoord]
        
    def format_perturbations(self):
        ## Now, find the zeros in ids: 
        self.to_remove = [np.where(self.ids_trimmed[:,i]==0) for i in range(2)] 
        ## Now, find the places where we should steal the other trajectory (ignore null points)
        ## This should monitor the presenc or absence of the current index, because it has already been flipped in the previous method.
        self.to_switch = [np.where((1-self.flips[:,i])) for i in range(2)]
        
        ##Now determine relevant intervals for each mouse's breakpoints and switchpoints. 
        self.breakpoint_bounds = [self.intervals_trimmed[self.to_remove[i],:].reshape(len(self.to_remove[i][0]),2) for i in range(2)]
        self.switchpoint_bounds = [self.intervals_trimmed[self.to_switch[i],:].reshape(len(self.to_switch[i][0]),2) for i in range(2)]
        ## Fill in intervals: 
        self.breakpoints = [list(np.concatenate([[]]+[np.arange(bound[i][0],bound[i][1]) for i in range(len(bound))]).astype(int)) for bound in self.breakpoint_bounds]
        self.switchpoints = [list(np.concatenate([[]]+[np.arange(bound[i][0],bound[i][1]) for i in range(len(bound))]).astype(int)) for bound in self.switchpoint_bounds]
        self.keepintervals = [self.intervals_trimmed[np.where(self.ids_trimmed[:,i]==1)] for i in range(2)]
        ## Additionally, just have lists of the points that are given in the presegments and the 
        self.prepoints = [np.concatenate([np.arange(self.presegments[i][j][0],self.presegments[i][j][1]) for j in range(len(self.presegments[i]))]) for i in range(2)]
        self.keeppoints = [np.concatenate([np.arange(self.keepintervals[i][j][0],self.keepintervals[i][j][1]) for j in range(len(self.keepintervals[i]))]) for i in range(2)]
        ## Finally, we want to have an easy lookup of which points have what ids. An array of 0,1 and nan, that indicates the identity of the point, should it exist, and nan if not.  
        index0,index1 = np.zeros(self.length),np.ones(self.length)
        index = np.stack((index0,index1),axis = 1)
        bindex = [np.stack((self.breakpoints[i],np.ones(len(self.breakpoints[i]))*i),axis = 1) for i in range(2)]
        all_bindex = np.concatenate(bindex,axis = 0).astype(int)
        index[all_bindex[:,0],all_bindex[:,1]] = np.nan
        sindex = [np.stack((self.switchpoints[i],np.ones(len(self.switchpoints[i]))*i),axis = 1) for i in range(2)]
        all_sindex = np.concatenate(sindex,axis = 0).astype(int)
        index[all_sindex[:,0],all_sindex[:,1]] = np.fliplr(index)[all_sindex[:,0],all_sindex[:,1]]
        self.idindex = index
        idsegs = np.zeros(self.ids_trimmed.shape)
        idsegs[:] = self.ids_trimmed
        idsegs[idsegs == 0] = np.nan
        self.idsegs = idsegs*self.flips
        self.idsegs[:,0] = abs(1-self.idsegs[:,0])

    ## We want to check that for any instance: 
    ## 1. Presegment points are in kept points
    ## 2. breakpoints are disjoint from kept points
    ## 3. switch points are disjoint from breakpoints of the other trajectory, and the current trajectory.  
    def check_points(self):
        cond1 = [np.all([j in a.keeppoints[i] for j in a.prepoints[i]]) for i in range(2)]
        cond2 = [np.any([j in a.keeppoints[i] for j in a.breakpoints[i]]) for i in range(2)]
        cond3 = [np.any([j in a.switchpoints[i] for j in a.breakpoints[i]]) for i in range(2)]
        cond4 = [np.any([j in a.switchpoints[abs(1-i)] for j in a.breakpoints[i]]) for i in range(2)]
        print(cond1,'True')
        print(cond2,'False')
        print(cond3,'False')
        print(cond4,'False')
    
    ## Plot both trajectories
    def check_plot(self):
        fig,ax = plt.subplots(2,)
        ax[0].plot(self.trajectory[:,0:2],'o')
        ax[1].plot(self.trajectory[:,2:4],'o')
        [[ax[i].axvline(x = p,color = 'r') for p in self.breakpoints[i]] for i in range(2)]
        [[ax[i].axvline(x = p,color = 'b') for p in self.switchpoints[i]] for i in range(2)]
        plt.show()

################### Distance Functions. 
    ## Give the distance between any two points in the unaltered data.
    ## This function takes in two tuples, representing the times and identities of two points. It returns a single scalar that represents the ground truth distance between the coordinates: i.e., excluding switches and breaks.  

    ## This is the linear interpolation distance between two points on the same circle. Suffices to give distance when considering an interpolation. 
    def gt_dist(self,times): 
        return 2*self.Circle.radius*abs(np.sin(np.pi*(times[1]-times[0])/float(self.Circle.period)))

    ## This is the distance around the circle (quantized perimeter parts). This is the bread and butter distance we will use when considering segments of keeppoints.   
    def gt_tv(self,times):
        return 2*self.Circle.radius*(times[1]-times[0])*abs(np.sin(np.pi/float(self.Circle.period)))

    ## Give the vector pointing from point at time a on a circle to point at time b on the same circle. 
    def vec(self,times):
        a,b = times[0],times[1]
        vec =  self.Circle.trajraw[b,0:2]-self.Circle.trajraw[a,0:2]
        return vec

    ## Give the distance between two id'd points on either circle. This is the last function we need in order to cover all possible distance cases. This case covers if there is a switch, and breakpoints on either side of the switch. Let's nail it! 
    def dist(self,times,ids):
        ## The approach is this. If both points are the same id, just give gt_dist. If not, calculate the distance to point b on the same circle. From point b, find vectors back to point a, and to point b on the other circle. Use these to solve for the distancee from a to b on the other circle with the law of cosines.   
        ## Find the true ids. 
        id0,id1 = self.idindex[times,ids]

        if id0 == id1:
            distance = self.gt_dist(times)

        else:
            ## Vector to point on same circle: 
            chordvec = self.vec([times[1],times[0]])
            ## Vector to point on other circle:  
            ## TODO: pass ids through the actual ids_trimmed for these points. 
            switchvec = self.Circle.centers[int(id1)]-self.Circle.centers[int(id0)]
            ## Get angle: 
            a = np.linalg.norm(chordvec)
            b = np.linalg.norm(switchvec)
            if a == 0 or b == 0:
                cosangle = 0 
            else:
                cosangle = np.dot(chordvec,switchvec)/(a*b)
            ## Law of Cosines: 
            a = np.linalg.norm(chordvec)
            b = np.linalg.norm(switchvec)
            distance = np.sqrt(a**2+b**2-2*a*b*cosangle)
        return distance

    ## Give the inter and intra interval distances (all of them)
    ## Inter interval distances should have the same signature as val_dist, minus intervals and mask (these are already given as object attributes) 
    ## Currind and mouseind are given in terms of id_trimmed segments. 
    ## We need a function to work with the interval and id construction. 
    ## Said function should use mask arrays to search the whole set of intervals for contiguous ones.
    ## Returns: two lists consisting of the valid indices arranged in a contiguous sequence. Then, once you give a certain array, we can easily find the ones that are before and after it in the list. 
    def construct_valids(self):
        self.valids = [np.arange(len(self.ids_trimmed))[np.where(self.ids_trimmed[:,i])] for i in range(2)]

    ## Recreate the val_dist functions we have in the Branch and Bound formulation.  
    def val_dist(self,currind,currid,mouseind,mouseid):
        ## Check if valid: 
        if np.isnan(self.idsegs[currind,currid]):
            dist = np.nan
        else:
            end = self.intervals_trimmed[mouseind,-1]-1
            start = self.intervals_trimmed[currind,0]
            dist = self.dist([int(end),int(start)],[mouseid,currid])
        return dist

    def val_time(self,currind,currid,mouseind,mouseid):
        ## Check if valid: 
        if np.isnan(self.idsegs[currind,currid]):
            dist = np.nan
        else:
            end = self.intervals_trimmed[mouseind,-1]-1
            start = self.intervals_trimmed[currind,0]
        return start-end 

    

    ## Now, get the intra segment distances, tv distances, and times.
    ## These are meant to correspond to actual trajectories, so their semantics are as follows: 
    ## 1. for filled segments, you get what you asked for. 
    ## 2. for nan segments, you will get the measurement on the linear interpolation. 
    ## 3. Times will generate whatever you asked for. 
    def intra_dist(self,i,m):
        ## Within a single segment, there should never be any subsegments with a different identity. This is why this funcion is well defined. 
        seg = [self.intervals_trimmed[i,0],self.intervals_trimmed[i,-1]-1]
        dist = self.gt_dist(seg) 
        return dist
    
    def intra_tv(self,i,m):
        seg = [self.intervals_trimmed[i,0],self.intervals_trimmed[i,-1]-1]
        if np.isnan(self.idsegs[i,m]):
            dist = self.gt_dist(seg) 
        else:
            ## Within a single segment, there should never be any subsegments with a different identity. This is why this funcion is well defined. 
            dist = self.gt_tv(seg) 
        return dist

    def intra_time(self,i,m):
        time = self.intervals_trimmed[i,-1]-self.intervals_trimmed[i,0]
        return time
            

    ## Now, calculate val dist over all neighboring pairs of valid trajectories. 
    def all_inter_dists(self):
        ## For each trajectory:
        dists = []
        for m in range(2):
            trajdists = []
            valids = self.valids[m]
            valids_pair = [[valids[i],valids[i+1]] for i in range(len(valids)-1)]
            for pair in valids_pair:
                trajdists.append(self.val_dist(pair[1],m,pair[0],m)) 
            dists.append(trajdists)
        return dists 

    def all_intra_tv(self):
        ## For each trajectory:
        dists = []
        for m in range(2):
            trajdists = []
            valids = self.valids[m]
            for valid in valids:
                trajdists.append(self.intra_tv(valid,m)) 
            dists.append(trajdists)
        return dists 
    
    ## Now, calculate the cost for both trajectories. If begin is given as a parameter, calculate two quantities, the cost up until that interval, and its complement.
    ## Validated for one switch, one break against the empirical calculations.  
    def return_tvcost(self,begin = 0):
        intercosts = self.all_inter_dists()
        intracosts = self.all_intra_tv()
        ## These are indexed by valid segments, and gaps between valid segments. Convert the begin index into the relevant indices for each mouse's valid indices. 
        geq  = [np.where(self.valids[i]>=begin)[0][0] for i in range(2)]
        ## Get the intercosts for this minimum segment and all greater. 

        inter_pre = [intercosts[i][:geq[i]] for i in range(2)]
        inter_post = [intercosts[i][geq[i]:] for i in range(2)]
        intra_pre = [intracosts[i][:geq[i]] for i in range(2)]
        intra_post = [intracosts[i][geq[i]:] for i in range(2)]
        
        ## Unroll all of these: 
        presum = np.sum(np.sum(inter_pre)) + np.sum(np.sum(intra_pre))
        postsum = np.sum(np.sum(inter_post)) + np.sum(np.sum(intra_post))
        return presum,postsum

    ## Now, add in the regularization terms: 
    # 1. removed indices: 
    def return_missingcost(self,begin = 0,lagrange = 1):
        # find locations in index space without data. 
        weightarray = np.zeros(np.shape(self.ids_trimmed))
        weightarray[np.where(self.ids_trimmed == 0)] = 1
        weights = np.sum(weightarray,axis = 1)
        ## The *number* of indices flipped in an interval is equal to the end minus start -1. 
        lengths = self.intervals_trimmed[:,1]-self.intervals_trimmed[:,0]
        costvec = lengths*weights*lagrange
        return np.sum(costvec[begin:])

    def return_switchcost(self,begin = 0,lagrange = 1): 
        weightarray = np.zeros(np.shape(self.ids_trimmed))
        weightarray[np.where(self.flips == 0)] = 1
        weights = np.sum(weightarray,axis = 1)
        lengths = self.intervals_trimmed[:,1]-self.intervals_trimmed[:,0]
        costvec = lengths*weights*lagrange
        return np.sum(costvec[begin:])

    ## TDOO: Calculate lagrange multipliers from the data: 


    ## Give the rates, and switch costs. 
    ## Give the cost for the unaltered route, the breakpoints only route, and the switches only route. 
    ## set things up so that ids_trimmed, flips, can be altered and the circle correspondingly redrawn. 
    ## 


## Now we want a function to check functions we have written against this framework.  

def test_val_funcs(func):
    ## First generate data: 
    mock = SwitchCircle(1000,40,30,0.7)
    ## Get out the raw trajectories (no nans): 
    trajraw = mock.Circle.return_switched()
    intervals = mock.intervals_trimmed.astype(int)
    mask = mock.ids_trimmed
    ## Now we will test the val function on valid interval differences.  
    alldiffs = []
    ## Initialize the functions we are testing as lambda functions: 
    if func == 'dist':
        testfunc = lambda w,x,y,z: val_dist(trajraw,intervals,mask,w,x,y,z)
        gtfunc = lambda w,x,y,z: mock.val_dist(w,x,y,z)
    if func == 'time':
        testfunc = lambda w,x,y,z: val_time(intervals,mask,w,x,y,z)
        gtfunc = lambda w,x,y,z: mock.val_time(w,x,y,z)
    ## Test the default configuration
    for m in range(2):
        valid = mock.valids[m]
        val_pairs = [[valid[i],valid[i+1]] for i in range(len(valid)-1)]
        diffs = np.ones(len(val_pairs))
        for v,val_pair in enumerate(val_pairs):
            test = testfunc(val_pair[1],m,val_pair[0],m)
            gt = gtfunc(val_pair[1],m,val_pair[0],m)
            diff = abs(test-gt)
            condition = 1-np.any(diff > 1e-10)
            diffs[v] = condition
        alldiffs.append(diffs)
    ## Test the flipped configuration
    for m in range(2):
        valid = mock.valids[m]
        other = mock.valids[abs(1-m)]
        diffs = np.ones(len(valid))
        for v,valind in enumerate(valid):
            ## We have to find the last valid point in the other trajectory!
            v_prev = other[np.where(other<valind)[0]]
            if len(v_prev) > 0:
                val_pair = [v_prev[-1],valind]
                test = testfunc(val_pair[1],m,val_pair[0],abs(1-m))
                gt = gtfunc(val_pair[1],m,val_pair[0],abs(1-m))
                diff = abs(test-gt)
                condition = 1-np.any(diff > 1e-10)
                diffs[v] = condition
        alldiffs.append(diffs)
    return alldiffs

## Like the above, but for intra functions. 
def test_intra_funcs(func):
    ## First generate data: 
    mock = SwitchCircle(1000,40,30,0.7)
    ## Get out the raw trajectories (no nans): 
    trajraw = mock.Circle.return_switched()
    intervals = mock.intervals_trimmed.astype(int)
    mask = mock.ids_trimmed
    ## Now we will test the val function on valid interval differences.  
    ## Initialize the functions we are testing as lambda functions: 
    if func == 'tv':
        testfunc = lambda x,y: intra_tv(trajraw,intervals,mask,x,y)
        gtfunc = lambda x,y: mock.intra_tv(x,y)
    if func == 'dist':
        testfunc = lambda x,y: intra_dist(trajraw,intervals,mask,x,y)
        gtfunc = lambda x,y: mock.intra_dist(x,y)
    if func == 'time':
        testfunc = lambda x,y: intra_time(intervals,mask,x,y)
        gtfunc = lambda x,y: mock.intra_time(x,y)
    ## This is simpler than the inter functions, because it does not rely on pairs. 
    all_diffs = []
    for m in range(2):
        ## If tv, we will only go through the valids:
        f = mock.valids[m]
        ## Find the lengths of each. 
        diffs = []
        for fi in f:
            test = testfunc(fi,m)
            gt = gtfunc(fi,m)
            diff = abs(test-gt)
            diffs.append(diff<1e-10)
        all_diffs.append(diffs)
    
    return all_diffs

def test_calculate_straight_cost():
    ## First generate data: 
    mock = SwitchCircle(1000,40,30,0.7)
    ## Get out the raw trajectories (no nans): 
    trajraw = mock.Circle.return_switched()
    intervals = mock.intervals_trimmed.astype(int)
    mask = mock.ids_trimmed
    signature = np.ones(np.prod(np.shape(obj.ids_trimmed)))
    partarray = [trajraw,intervals,mask]
    z = calculate_straight_cost(partarray,signature)
    return z




    
    
    

    


