from __future__ import division
from collections import defaultdict
from operator import itemgetter
import operator
import scipy.io
from collections import Counter
SB=[29348,344538,412928,523387,235719,466535,34678,406001,520119,520628,478023,466547,66242,347464,104994,236251,89195,165321,499215,466838]
wp=defaultdict(int)
PapCitPatDict = defaultdict(lambda : defaultdict(int))
CitYrsDict = defaultdict(list)

def preparedata():
	global wp
	dataset = int(raw_input( "Enter 1 for CS and 0 for Pubmed : "))
	if dataset == 1:
		print "\nLoading Files"
		PapCitPattern = scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPapCitPattern.mat')
		PapCitPattern = PapCitPattern['CsPapCitPattern']
	else:
		PapCitPattern = scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/codes/matlab-codes/PapCitPattern.mat')
		PapCitPattern = PapCitPattern['PapCitPattern']
	f=open('/home/siddharth/Desktop/PROJECT/ERP/csmag/WakeupPoint/CsMyWakePeriodNew_11.txt',"r")

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

	for l in f:
		l=l.strip().split()
		if int(l[0])!=478023:
			wp[int(l[0])]=int(l[2])+1


def peakStats(S):
	T = wp[S]

	citPattern=PapCitPatDict[S]
	MaxCitYr = max(citPattern.iteritems(), key=operator.itemgetter(1))[0]
	PeakingPeriod = MaxCitYr-T+1
	
	PeakingCit = sum(citPattern[y] for y in citPattern if y in range(T,MaxCitYr+1))
	PeakingRate = PeakingCit/PeakingPeriod

	PeakStats = [S,T,MaxCitYr,PeakingCit,PeakingRate,PeakingPeriod]

	return PeakStats


#-------------------------------------Main code---------------------------------------

preparedata() # Function To Prepare necessary dictioanries
SBStats=[]
for ID in SB:
	#ID = int(raw_input("Enter ID of a sleeping beauty paper : "))
	#a =  float(raw_input("Enter Citation Count Parameter : "))
	#b =  float(raw_input("Enter Citation Rate Parameter : "))
	SBPeakStat = peakStats(ID)
	SBStats.append([SBPeakStat[0],SBPeakStat[1],SBPeakStat[2],SBPeakStat[3],SBPeakStat[4],SBPeakStat[5]])
	print "Peaking Statistics For %d : "%(ID)
	print SBPeakStat

f=open('/home/siddharth/Desktop/PROJECT/ERP/csmag/WakeupPoint/CsSBPeakingStatsNew_11.txt','w')
line = ""
for l in SBStats:
	line += str(l[0])+"\t"+str(l[1])+"\t"+str(l[2])+"\t"+str(l[3])+"\t"+str(l[4])+"\t"+str(l[5])+"\n"
f.write(line)
f.close()



