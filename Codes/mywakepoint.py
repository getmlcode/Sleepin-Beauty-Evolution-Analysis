from __future__ import division
from collections import defaultdict
import matplotlib.pyplot as plt
from operator import itemgetter
import operator
import scipy.io
from collections import Counter
import numpy
import random
import math

#SB=[29348,344538,412928,523387,235719,466535,34678,406001,520119,520628,478023,466547,66242,347464,104994,236251,89195,165321,499215,466838]
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


def GetWakeTime(S,alpha,beta):
	citPattern = PapCitPatDict[S]
	citYrs = CitYrsDict[S]
	
	print "Citing Years for %d : "%(S)
	print citYrs
	MaxCitYr = max(citPattern.iteritems(), key=operator.itemgetter(1))[0]
	MaxCit = citPattern[MaxCitYr]
	print "Max Cit = %d Received in %d"%(MaxCit,MaxCitYr)
	
	PubYr = PapYrDict[S]
	YrCit = []
	for y in range(MaxCitYr-PubYr+1):
		T = y + PubYr
		if T in citYrs:
			Cit = citPattern[T]
		else:
			Cit = 0	
		YrCit.append([T,Cit])

	print "Year Citation : ",YrCit

	YrCit = sorted(YrCit,key=itemgetter(0))
	Yr = [e[0] for e in YrCit]
	Cit = [e[1] for e in YrCit]

	plt.xlabel("Year")
	plt.ylabel("Citations")
	plt.plot(Yr,Cit,'ro-')
	plt.show()
	plt.close()

	if PubYr in citYrs:
		PubYrCit = citPattern[PubYr]
	else:
		PubYrCit = 0	
	print "Cit in Published Year %d = %d"%(PubYr,PubYrCit)

	WakeYr = PubYr

	TransYr = int(raw_input("Enter transition year : "))
	TrDuration = TransYr - PubYr
	TotPriorCit = sum(citPattern[y] for y in citPattern if y < TransYr)
	PriorCitRate = TotPriorCit/TrDuration

	print "Total Prior Citation : ",TotPriorCit
	print "Prior Citation Rate : ",PriorCitRate
	
	nextYrs = sorted([k for k in citPattern if k>=TransYr])
	print "next yeras : ",nextYrs

	WakeCit = 0
	WakeRate = 0
	for y in nextYrs:
		WakeCit += citPattern[y]
		time = y - TransYr+1
		WakeCitRate = WakeCit/time
		print "wake cit : ",WakeCit
		print "wake rate : ",WakeCitRate
		if WakeCit > TotPriorCit*alpha and WakeCitRate > PriorCitRate*beta:
			print "wake cit : ",WakeCit
			print "wake rate : ",WakeCitRate	
			WakeYr = y
			break
	return WakeYr


#-------------------------------------Main code---------------------------------------


preparedata() # Function To Prepare necessary dictioanries

option=1
while option!=0:
	ID = int(raw_input("Enter ID of a sleeping beauty paper : "))
	a =  float(raw_input("Enter Citation Count Parameter : "))
	b =  float(raw_input("Enter Citation Rate Parameter : "))
	WakeTime = GetWakeTime(ID,a,b)

	print "Wake Up Time For %d is %d"%(ID,WakeTime)
	option = int(raw_input("Enter Zero To Stop : "))
