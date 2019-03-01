from collections import defaultdict
import numpy as np
import scipy.io
import time


PapCitPattern=scipy.io.loadmat('E:\\bin\\PapCitPattern.mat')
PubYrCit=scipy.io.loadmat('E:\\bin\\PubYrCit.mat')
p_year=scipy.io.loadmat('E:\\bin\\p_year.mat')

p_year=p_year['p_year']
PapCitPattern=PapCitPattern['PapCitPattern']
PubYrCit=PubYrCit['PubYrCit']

ppid=PapCitPattern[:,:1]                                    #Extracting First Column
ppid=np.unique(ppid)                                        #Retaining Unique Values  
btcoff=[[0,0]]
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
    
    bty =0

    time = maxyr-pubyr
    b=maxcit-pubcit
    slope = b/(time*1.0)

    if time > 3:
        for j in range(1,time+1):
            yr = pubyr+j-1
            if yr in CitYrs:
                ct=CitPattern[CitPattern[:,1]==yr,2]
                d=ct
            else:
                ct=0
                d=1
            lt=slope*(j-1)+pubcit
            inc=(lt-ct)/(d*1.0)
            bty=bty+inc

    b=[id,bty]
    b=np.array([b])
    btcoff=np.concatenate((btcoff,b),axis=0)
    print "beauty coefficient of paper %d is %f"%(id,bty)

btcoff = btcoff[1:,:]
scio.savemat('F:\\Python27\\mycodes\\btycoff.mat',{'btcoff':btcoff})

    
