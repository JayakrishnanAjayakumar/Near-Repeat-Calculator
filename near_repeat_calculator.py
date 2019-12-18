from forest import *
from forest.bobs.Bobs import *
import forest.engines.Config
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import sharedctypes as sct
import ctypes
import binpacking
from numba import jit
from scipy.stats import rankdata
from collections import OrderedDict,deque
import math
from shapely.geometry import box
import argparse

parser = argparse.ArgumentParser(description='Nearrepeat calculator')
parser.add_argument("--distancebandinmeters", default=121.92,type=float, help="spatial band")
parser.add_argument("--timebandindays", default=14,type=int, help="timeband")
parser.add_argument("--maximumdistmeters", default=1219.2,type=float, help="maximum distance up to which to calculate near repeat patterns")
parser.add_argument("--maxdays", default=70,type=int, help="maximum day up to which to calculate near repeat patterns")
parser.add_argument("--kstack", default=10,type=int, help="k-stack size")
parser.add_argument("--numproc", default=1,type=int, help="Number of processes")
parser.add_argument("--totalsim", default=99,type=int, help="Number of Simulations (generally 99,999,9999 etc)")
parser.add_argument("--numblocks", default=256,type=int, help="Number of subdomains. Should be a power of 2 and more than number of processors eg 16,64,256 etc")
parser.add_argument("--filepath", default=1,type=str,required=True, help="Full file path eg /home/jajayaku/crimeset.csv")
args = parser.parse_args()

#distance band, time band, maximum distance and maximum time for near repeat caclulator
distancebufferinmeters,timebufferinmilliseconds,maxdist,maxtime=args.distancebandinmeters,args.timebandindays*24*60*60*1000,args.maximumdistmeters,args.maxdays*24*60*60*1000
#simulation count
totalsim=args.totalsim
#create the ranges and bins
distancebins=np.append(np.linspace(0,maxdist,num=(maxdist/distancebufferinmeters)+1),np.inf)+1
distancebins=np.append(np.asarray([0]),distancebins)
timebins=np.append(np.linspace(0,maxtime,num=(maxtime/timebufferinmilliseconds)+1),np.inf)
timebins[1:]+=24*60*60*1000
#number of processors
p=args.numproc
#numtiles to split the bobs
numblocks=args.numblocks
#how many time slices to store (K)
itersetsize=args.kstack
#blockdepth for slices
blockdepth=np.minimum(itersetsize,totalsim+1)
#size of final matrix in each processor
processoroutsize=(totalsim+1)*(len(distancebins)-1)*(len(timebins)-1)

#fast numba based distance calculation. Faster than cdist offered by scipy spatial
@jit(nopython=True)
def cdist_numba(point,pointarr,outarr):
    for i in range(len(pointarr)):
        outarr[i]=np.sqrt(((point[0]-pointarr[i,0])**2)+((point[1]-pointarr[i,1])**2))

#numba based matrix filling
@jit(nopython=True)
def numba_fill(depthval,timebinvalsarray,distbinvals,outmatslice):
    for depth in range(depthval):
        timedat=timebinvalsarray[depth,]
        for k in range(len(timedat)):
            outmatslice[depth,distbinvals[k],timedat[k]]+=1

#near repeat method. Accepts a bob datastructure, linear time array, outmatrix to write to, start index for outmatrix filling, end index for outmatrix filling          
def nearrepeat(bob,timearray,outmat,indstart,indend):
    #get all internal x,y into an array
    bobpoints=np.asarray([[dat['x'],dat['y']] for dat in bob.data])
    #for all the internal points exluding the halo, perform pairwise distance and time calculation
    for i in range(len(bobpoints)-1):
        #the distance value will be stored in this array
        distarr=np.zeros(len(bobpoints[i+1:]))
        #using numba for speedups
        cdist_numba(bobpoints[i],bobpoints[i+1:],distarr)
        #bin the values
        distbinvals=np.digitize(distarr,distancebins)-1
        #get the time differences which will be used for calculations
        timedataarr=np.absolute(timearray[0:(indend-indstart),:,i]-timearray[0:(indend-indstart),0,i+1:len(bobpoints)])
        #bin the time differences. This will be a multidimensional matrix
        timebinvalsarray=np.digitize(timedataarr,timebins)-1
        numba_fill(indend-indstart,timebinvalsarray,distbinvals,outmat[indstart:indend,:,:])
    #get all halo data into an array
    neighbpoints=np.asarray([[dat['x'],dat['y']] for dat in bob.halo])
    #perform pairwise calculation between halo and internal points
    if len(neighbpoints!=0):
        for i in range(len(bobpoints)):
            #the distance value will be stored in this array
            distarr=np.zeros(len(neighbpoints))
            #using numba for speedups
            cdist_numba(bobpoints[i],neighbpoints,distarr)
            #bin the values
            distbinvals=np.digitize(distarr,distancebins)-1
            #get the time differences which will be used for calculations
            timedataarr=np.absolute(timearray[0:(indend-indstart),:,i]-timearray[0:(indend-indstart),0,len(bobpoints):])
            #bin the time differences. This will be a multidimensional matrix
            timebinvalsarray=np.digitize(timedataarr,timebins)-1
            numba_fill(indend-indstart,timebinvalsarray,distbinvals,outmat[indstart:indend,:,:])
               
'''Worker function to be executed by each worker (process). Accepts a list of bobs, transferbuffer to store permuted time recieved from master, statusarray to notify master about data transfer,
,A global countervar which needs to be updated after every recieval of time data, the id for the process, lock for writing to shared structures, an output knox table, completionarray to indicate the completion of all tasks,
total indicating the number of values to copy from transfer array
The bobworker calculates a complete knox table with dimensions (simcount+1,space,time) with the list of bobdata provided to it. The time permutations for each simulations are recieved through the shared datastructure transferbuffer'''

def bobworker(bob_data,transferbuffer,statusarray,countervar,myid,lock,outmat_raw,completionarray,total):
    starttime=timer()
    #reshape the buffer to change it into a multidimensional matrix
    outmat=np.frombuffer(outmat_raw,dtype=np.int32).reshape((totalsim+1),(len(distancebins)-1),(len(timebins)-1))
    timedatastore=[]
    for bob in bob_data:
        storage=np.zeros((blockdepth,1,len(bob.data)+len(bob.halo)),dtype=long)
        alldat=[]
        for b in bob.data:
            alldat.append(long(b['t']))
        for h in bob.halo:
            alldat.append(long(h['t']))
        storage[0,:,:]=alldat
        timedatastore.append(storage)
    #how many iterations are processed
    localiterations=0
    #for the first set we have observed so setting the tempitercount to 1
    tempitercount=1
    while localiterations<totalsim+1:
        #the blockdepth for this iteration set, if the iterations left is less than block depth we select that. This is essentially K stack
        thisiter=np.minimum(blockdepth,totalsim+1-localiterations)
        #loop until we reach the required number of iterations
        while tempitercount<thisiter:
            #with lock access the resource for data transfer
            timedata=None
            data=[]
            for bob in bob_data:
                data.extend([dat['id'] for dat in bob.data])
                data.extend([dat['id'] for dat in bob.halo])
            with lock: 
                transferbuffer[0:total]=data
                statusarray[myid]=1
                while statusarray[myid]!=2:
                    pass
                #once we have timedata release the lock and start processing the timedata
                timedata=np.asarray(transferbuffer[0:total])
                countervar.value=countervar.value+1
            data=None
            if timedata is not None:
                j=0
                for i in range(len(bob_data)):
                    timedatastore[i][tempitercount,:,:]=timedata[j:j+len(bob_data[i].data)+len(bob_data[i].halo)]
                    j+=len(bob_data[i].data)+len(bob_data[i].halo)
                tempitercount+=1
            #need to check if all processors got there data for this iteration
            while countervar.value%p!=0:
                pass
        for i in range(len(bob_data)):
            nearrepeat(bob_data[i],timedatastore[i],outmat,localiterations,localiterations+tempitercount)
        #update local iterations
        localiterations+=tempitercount
        #update temp iteration count
        tempitercount=0
    print 'time taken = '+str(timer()-starttime)
    completionarray[myid]=1

#calculate rank of bserved among simulations       
def rank_dat(arr):
    return (1+len(arr))-rankdata(arr,method='min')[0]

#calculating knox ratio based on observed/median of simualted
def calc_knox_ratio(arr):
    knox_rat=0.0
    median_v=np.median(arr[1:])
    if median_v!=0:
        knox_rat= float(arr[0])/median_v
    return knox_rat

#split a layer horizontally based on midpoint of x. The out is two layers.
def x_split(layer):
    layer_out=[]
    #sort by X
    x_sorted=sorted(layer.data,key=lambda x:x['x'])
    #partition index which is the midpoint
    partition_x_index=int(len(x_sorted)/2.0)
    layer1=STPoint(layer.y,layer.x, layer.h,x_sorted[partition_x_index-1]['x']-layer.x, layer.s, layer.d)
    layer2=STPoint(layer.y,x_sorted[partition_x_index]['x'], layer.h,layer.x+layer.w-x_sorted[partition_x_index]['x'], layer.s, layer.d)
    layer1.data=np.asarray(x_sorted[0:partition_x_index])
    layer2.data=np.asarray(x_sorted[partition_x_index:])
    layer_out.append(layer1)
    layer_out.append(layer2)
    return layer_out

#split a layer vertically based on midpoint of y. The out is two layers.
def y_split(layer):
    layer_out=[]
    #sort by Y
    y_sorted=sorted(layer.data,key=lambda x:x['y'])
    #partition index which is the midpoint
    partition_y_index=int(len(y_sorted)/2.0)
    layer1=STPoint(layer.y,layer.x, y_sorted[partition_y_index-1]['y']-layer.y,layer.w, layer.s, layer.d)
    layer2=STPoint(y_sorted[partition_y_index]['y'],layer.x, layer.y+layer.h-y_sorted[partition_y_index]['y'],layer.w, layer.s, layer.d)
    layer1.data=np.asarray(y_sorted[0:partition_y_index])
    layer2.data=np.asarray(y_sorted[partition_y_index:])
    layer_out.append(layer1)
    layer_out.append(layer2)
    return layer_out

#read a csv file                    
def testreadcsv(csvfilename):
    csvlayer=CsvRead(filename=csvfilename)
    return csvlayer

#find indexes for a sorted array based on left and right bounds. Uses binary search to find the indexes.
def getbinindexes(array,bounds,key):
    start,end=0,len(array)
    right,left=None,None
    possible_start=None
    if bounds[0]>array[len(array)-1][key] or bounds[1]<array[0][key]:
        return [0,0]
    while start<end:
        midindex=(end-start)/2
        if array[start+midindex][key]<=bounds[1]:
            if midindex==0:
                right=start+1
                break
            start=start+midindex
            possible_start=start
        else:
            if midindex==0:
                right=None
                break
            end=start+midindex
            if possible_start is not None:
                for i in range(possible_start,end+1):
                    if array[i][key]>bounds[1]:
                        right=i
                        break
                break
    start,end=0,len(array)
    possible_end=None
    while start<end:
        midindex=(end-start)/2 
        if midindex==0:
            if array[start][key]>=bounds[0]:
                left=start
                while left>=1:
                    if array[left-1][key]==array[start+midindex][key]:
                        left-=1
                    else:
                        break
            else:
                if array[start+1][key]>=bounds[0]:
                    left=start+1
            break
        if array[start+midindex][key]>bounds[0]:
            end=start+midindex
            possible_end=end
        else:
            start=start+midindex
            if possible_end is not None:
                for i in range(start,possible_end+1):
                    if array[i][key]>=bounds[0]:
                        left=i
                        break
                exist_left=left
                while left>=1:
                    if array[left-1][key]==array[exist_left][key]:
                        left-=1
                    else:
                        break
                break
    if left is None or right is None:
        return [0,0]
    return [left,right]
            
if __name__ == '__main__':
    #read the main layer bob
    mainlayer=testreadcsv(args.filepath)
    totaldatasize=len(mainlayer.data)
    grainsize=totaldatasize/float(numblocks)
    #check threshold
    if grainsize<2:
        while grainsize<2:
            numblocks/=2
            grainsize=totaldatasize/float(numblocks)
    #levels should be split based on numblocks
    splitlevels=int(math.log(numblocks,2))
    #do the DRAB to partition equally to p processors
    q=deque([mainlayer])
    for i in range(splitlevels):
        for j in range(2**i):
            layer=q.popleft()
            results=None
            if i%2!=0:
                results=x_split(layer)
            else:
                results=y_split(layer)
            q.append(results[0])
            q.append(results[1])
    alllayers=list(q)
    q=None
    #for each layers we need to add the halozones
    #The relations will be a mapping between the halo zones and the corresponding boxes of intersection
    relations=OrderedDict()
    for i in range(len(alllayers)-1):
        thislayer=alllayers[i]
        #we will use shapely to store this as a geometry to do easy intersection tests
        haloboundariesforthislayer=box(thislayer.x-maxdist,thislayer.y-maxdist,thislayer.x+thislayer.w+maxdist,thislayer.y+thislayer.h+maxdist)
        for j in range(i+1,len(alllayers)):
            otherlayer=alllayers[j]
            boundaryforotherlayer=box(otherlayer.x,otherlayer.y,otherlayer.x+otherlayer.w,otherlayer.y+otherlayer.h)
            if haloboundariesforthislayer.intersects(boundaryforotherlayer):
                if j not in relations:
                    relations[j]=OrderedDict()
                relations[j][i]=haloboundariesforthislayer.intersection(boundaryforotherlayer).bounds
    #take each contributing layer
    for indexes in relations:
        parent_layer=alllayers[indexes]
        #do an X sort
        sort_x=sorted(parent_layer.data,key=lambda x:x['x'])
        #take each childs get the corresponding bounds
        for childind in relations[indexes]:
            childbounds=relations[indexes][childind]
            if len(childbounds)!=4:
	        continue
            #this will get the start and end indexes for the x_sorted data based on bounds
            binindexes=getbinindexes(sort_x,[childbounds[0],childbounds[2]],'x')
            dat=sort_x[binindexes[0]:binindexes[1]]
            if len(dat)!=0:
                #do a y_sort get the y indexes
                sort_y=sorted(dat,key=lambda x:x['y'])
                binyindexes=getbinindexes(sort_y,[childbounds[1],childbounds[3]],'y')
                #add the halos
                alllayers[childind].halo.extend(sort_y[binyindexes[0]:binyindexes[1]])
    
    #bin packing for efficient load balancing
    forbinpacking=[(i,len(b.data)+len(b.halo)) for i,b in enumerate(alllayers)]
    #bin packing to evenly arrange the bobs
    bins = binpacking.to_constant_bin_number(forbinpacking,p,weight_pos=1)
    datineach=[]
    for b in bins:
        proc_datasize=0
        for tup in b:
            proc_datasize+=tup[1]
        datineach.append(proc_datasize)
    #shared trasnfer array for transferring time slices
    transferarray=sct.RawArray(ctypes.c_longlong,max(datineach))
    #shared array for this process
    out_array=sct.RawArray(ctypes.c_int,processoroutsize)
    # a list of shared array for all the processes
    out_shared_list=[out_array]*p
    completionarray=sct.RawArray(ctypes.c_int,p)
    completionarray[:]=[0]*p
    statusarray=sct.RawArray(ctypes.c_int,p)
    countervar=multiprocessing.Value('i', 0)
    jobs=[]
    #lock to access the buffer 
    lock=multiprocessing.Lock()
    #start the processes
    for pr in range(p):
        bobdata=[]
        for tup in bins[pr]:
            bobdata.append(alllayers[tup[0]])
        #start the processes
        process=multiprocessing.Process(target=bobworker, args=(bobdata,transferarray,statusarray,countervar,pr,lock,out_shared_list[pr],completionarray,datineach[pr]))
        process.start()
        jobs.append(process)
    #iteration dictionaty to keep track of the performed iterations
    iterdict={}
    #local iteration count till now
    locitcount=0
    #shuffled arrray
    shuffled=None
    #till the last processor is completed
    processor_catered_request=np.ones(p)
    #loop to generate time permutations
    while True:
        if sum(completionarray)==p:
            break
        #current status of all the processors
        currentval=countervar.value
        #if one cycle completed then we migh want a shuffling
        if currentval%p==0 and locitcount<totalsim:
            if(currentval not in iterdict):
                #random permutation of main layer data
                #print 'shuffling'
                shuffled=np.random.permutation(len(mainlayer.data))
                processor_catered_request[:]=0
                #update iteration dictionary
                iterdict[currentval]=0
                #update local iteration count
                locitcount+=1
                #print locitcount
        #look for requests and start catering the requests
        incomprocessors=np.where(np.asarray(completionarray)==0)
        for i in incomprocessors[0]:
            if statusarray[i]==1 and processor_catered_request[i]==0:
                #get the real indexes from the transferbufferlist as it will have the required ids
                realindexes=transferarray[0:datineach[i]]
                #get the timedata for the realindexes based on the shuffled array
                realdata=np.asarray([long(b['t']) for b in mainlayer.data[shuffled[realindexes]]],dtype=long)
                #pass it back to the buffer list
                transferarray[0:datineach[i]]=realdata
                statusarray[i]=2
                processor_catered_request[i]=1
    #final sum will be stored in this matrix
    finalsum=np.zeros(processoroutsize,dtype=int).reshape((totalsim+1),(len(distancebins)-1),(len(timebins)-1))
    #loop over and calculate the sum from each processes
    for proc in range(p):
        finalsum+=np.asarray(out_shared_list[proc]).reshape((totalsim+1),(len(distancebins)-1),(len(timebins)-1))
    #align the axis for calculating the rank of observed values
    aligned=np.dstack(finalsum).reshape((len(distancebins)-1)*(len(timebins)-1),totalsim+1)
    #apply the rank statisitc along the axis and reshape to create a 2d matrix with ranks. Divide by total simulation+observation to get p values
    out_final_p=1-(np.apply_along_axis(rank_dat, 1, aligned).reshape(len(distancebins)-1,len(timebins)-1)/float((totalsim+1)))
    print 'pvalues'
    print out_final_p
    knox_ratios=np.apply_along_axis(calc_knox_ratio, 1, aligned).reshape(len(distancebins)-1,len(timebins)-1)/float((totalsim+1))
    print 'knox_ratio'
    print knox_ratios
