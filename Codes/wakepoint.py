from __future__ import division
from collections import defaultdict
from operator import itemgetter
import operator
import scipy.io
from collections import Counter
import numpy
import random
import math

SB=[29348,344538,412928,523387,235719,466535,34678,406001,520119,520628,478023,466547,66242,347464,104994,236251,89195,165321,499215,466838]
PapCitPatDict = defaultdict(lambda : defaultdict(int))
CitYrsDict = defaultdict(list)
PapYrDict=defaultdict(int)
PapAuthDict=defaultdict(list)

def preparedata():
	dataset = int(raw_input( "Enter 1 for CS and 0 for Pubmed : "))
	if dataset == 1:
		print "\nLoading Files"
		PapCitPattern = scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPapCitPattern.mat')
		PapCitPattern = PapCitPattern['CsPapCitPattern']
		PapAuth=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
		PapYr=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidYr.txt')
	else:
		PapCitPattern = scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/codes/matlab-codes/PapCitPattern.mat')
		PapCitPattern = PapCitPattern['PapCitPattern']
		PapAuth=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
		PapYr=scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/codes/matlab-codes/p_year.mat')
		PapYr=PapYr['p_year']

	print "\nCreating paper-year Dictionary"
	for row in PapYr:
		PapYrDict[int(row[0])]=int(row[1])

	print "\nCreating paper-author Dictionary"
	for row in PapAuth:
		PapAuthDict[int(row[0])]+=[int(row[1])]

	print "\nCreating Citation Pattern and Citation Years Dictionaries"
	d=defaultdict(int)
	d={}
	i=1
	for row in PapCitPattern:
		if i !=row[0]:
		    d={}
		    i=row[0]
		d[row[1]]=row[2]
		PapCitPatDict[row[0]]=d
		CitYrsDict[row[0]]=d.keys()
	print "\nCitation Pattern Dictionary and Cit Years Dict Created"


def GetWakeTime(S):
	citPattern = PapCitPatDict[S]
	citYrs = CitYrsDict[S]
	
	print "Citing Years for %d : "%(S)
	print citYrs
	MaxCitYr = max(citPattern.iteritems(), key=operator.itemgetter(1))[0]
	MaxCit = citPattern[MaxCitYr]
	print "Max Cit = %d Received in %d"%(MaxCit,MaxCitYr)
	PubYr = PapYrDict[S]
	if PubYr in citYrs:
		PubYrCit = citPattern[PubYr]
	else:
		PubYrCit = 0
	
	print "Cit in Published Year %d = %d"%(PubYr,PubYrCit)

	wakeYr = PubYr
	MaxDist = 0
	mcy = MaxCitYr-PubYr 
	for t in range(MaxCitYr-PubYr):
		T = PubYr+t
		if T in citYrs:
			Ct = citPattern[T]
		else :
			Ct = 0
		n=abs( (MaxCit-PubYrCit)*t - mcy*Ct + mcy*PubYrCit)
		d=math.sqrt(( (MaxCit-PubYrCit)**2 + mcy**2 ))
		Dist = n/d
		print "Dist : ",Dist

		if Dist > MaxDist:
			print "Before Modification MaxDist = ",MaxDist
			MaxDist = Dist
			WakeYr = T
			print "MaxDist Modified for year ",T
			print "After Modification MaxDist = ",MaxDist

	return WakeYr


#-------------------------------------Main code---------------------------------------


preparedata() # Function To Prepare necessary dictioanries

option=1
PapWakTime=[]
for ID in SB:
	WakeTime = GetWakeTime(ID)
	print "Wake Up Time For %d is %d"%(ID,WakeTime)
	PapWakTime.append([ID,WakeTime])

f=open('/home/siddharth/Desktop/PROJECT/ERP/csmag/WakeupPoint/CsWakePoint.txt','w')
line = ""
for l in PapWakTime:
	line += str(l[0]) + "\t" + str(l[1])+"\n"
f.write(line)
f.close()




