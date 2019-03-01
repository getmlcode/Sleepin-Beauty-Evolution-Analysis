from collections import defaultdict
import numpy as np
import scipy.io
import time
import pp

job_server = pp.Server()

def B_coff(st,en,slope,CitPattern,pubyr,pubcit,CitYrs):
    bty=0
    for j in range(st,en+1):
            yr = pubyr+j-1
            #print yr
            if yr in CitYrs:
                ct=CitPattern[CitPattern[:,1]==yr,2]
                d=ct
            else:
                ct=0
                d=1
            lt=slope*(j-1)+pubcit
            inc=(lt-ct)/(d*1.0)
            bty=bty+inc
    return bty
    

PapCitPattern=scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CSMAG INT/CsPapCitPattern.mat')
PubYrCit=scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CSMAG INT/CsPubYrCit.mat')
p_year=scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CSMAG INT/CsPidYr.mat')

p_year=p_year['p_year']
PapCitPattern=PapCitPattern['PapCitPattern']
PubYrCit=PubYrCit['PubYrCit']

ppid=PapCitPattern[:,:1]                                    #Extracting First Column
ppid=np.unique(ppid)                                        #Retaining Unique Values  

for id in ppid:
    #id=938
    CitPattern = PapCitPattern[PapCitPattern[:,0]==id]      #conditionally Selecting Rows

    CitYrs = CitPattern[:,1]
    c=CitPattern[:,2:]
    i,j = np.unravel_index(c.argmax(), c.shape)             #Finding index for max value

    maxcit = CitPattern[i][2]
    maxyr  = CitPattern[i][1]
    
    pubyr = p_year[p_year[:,0]==id,1]
    pubcit = PubYrCit[PubYrCit[:,0]==id,1]
    
    btyC =0

    time = maxyr-pubyr
    b=maxcit-pubcit
    slope = b/(time*1.0)

    if time > 3:
        mid  = time/2
        job1 = job_server.submit(B_coff, (1,mid,slope,CitPattern,pubyr,pubcit,CitYrs))
        job2 = job_server.submit(B_coff, (mid+1,time,slope,CitPattern,pubyr,pubcit,CitYrs))
        btyL = job1()
        btyR = job2()
        
        btyC=btyL+btyR

    print "beauty coefficient of paper %d is %f"%(id,btyC)
    

