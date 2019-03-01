from __future__ import division
from collections import defaultdict
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
import numpy
import scipy.io



PapAuthDict=defaultdict(list)
AuthPapDict=defaultdict(list)

PapRefDict=defaultdict(list)
PapCitDict=defaultdict(list)

PapYrDict=defaultdict(int)
YrPapDict=defaultdict(list)
UptoYrPapDict=defaultdict(list)
UptoYrAuthDict=defaultdict(list)

PapCitCount=defaultdict(int)

FolLeadCoefDict={}
AuthRefHistDict={}
YrPapRankDict=defaultdict(int)
YrAuthRankDict=defaultdict(int)

YrNoOfAuth=0
YrNoOfPap=0
RefRank=0



def preparedata():
	print "\nLoading Files"
	PapAuth=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
	PapYr=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidYr.txt')
	PaperPaper=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidRefPid.txt')

	print "\nCreating paper-author and author-paper Dictionaries"
	for row in PapAuth:
		PapAuthDict[int(row[0])]+=[int(row[1])]
		AuthPapDict[int(row[1])]+=[int(row[0])]

	print "\nCreating reference and citations Dictionaries"
	for row in PaperPaper:
		PapRefDict[int(row[0])]+=[int(row[1])]


	print "\nCreating year-paper and paper-year Dictionaries"
	for row in PapYr:
		PapYrDict[int(row[0])]=int(row[1])
		YrPapDict[int(row[1])]+=[int(row[0])]
	print "\nDictionaries Created"

	'''print "Creating papers upto year dictionary"
	yrs=sorted(YrPapDict.keys())
	UptoYrPapDict[yrs[0]] = YrPapDict[yrs[0]]
	for i in range(1,len(yrs)):
		UptoYrPapDict[yrs[i]] = UptoYrPapDict[yrs[i-1]] + YrPapDict[yrs[i]]
	print "Papers upto year dictionary created"

	print "Creating Authors upto year dictionary"
	for yr in UptoYrPapDict:
		BeforeYrAuth=defaultdict(list)
		for p in UptoYrPapDict[yr]:
			for a in PapAuthDict[p]:
				BeforeYrAuth[a]+=[p]
		UptoYrAuthDict[yr] = BeforeYrAuth.keys()
	print "Authors upto year dictionary created" '''


def PlotAvgRank():
	#better auth and paps have higher index value(as read from file)

	Int1YrAvg=[]
	Int2YrAvg=[]
	Int3YrAvg=[]
	Int4YrAvg=[]
	Int5YrAvg=[]
	Int6YrAvg=[]
	

	for yr in range(1990,2016):
		print "Calculating Avg Rank for year %d " %(yr)
		p = "/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseAuthRanks/"+str(yr)+"AuthRank.txt"
		YrAuthRankFile = open(p,"r")
		for l in YrAuthRankFile:
			l=l.strip().split('\t')
			YrAuthRankDict[int(l[0])]=int(l[1])

		p = "/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWisePapRanks/"+str(yr)+"PapRank.txt"
		YrPapRankFile = open(p,"r")
		for l in YrPapRankFile:
			l=l.strip().split('\t')
			YrPapRankDict[int(l[0])]=int(l[1])
	
		YrPapers = YrPapDict[yr]

		YrAuthRankFile.close()
		YrPapRankFile.close()

		YrNoOfAuth = len(YrAuthRankDict)
		YrNoOfPap  = len(YrPapRankDict)

		Thresh1 = YrNoOfAuth-20
		Thresh2 = Thresh1 - 79
		Thresh3 = Thresh2 - 500
		Thresh4 = Thresh3 - 1000
		Thresh5 = Thresh3 - 2500

		YrAuthInt1 = [a for a in YrAuthRankDict if YrAuthRankDict[a]>Thresh1] #1-20
		YrAuthInt2 = [a for a in YrAuthRankDict if YrAuthRankDict[a]>=Thresh2 and YrAuthRankDict[a]<=Thresh1] #21-100
		YrAuthInt3 = [a for a in YrAuthRankDict if YrAuthRankDict[a]>=Thresh3 and YrAuthRankDict[a]<Thresh2] #101-500
		YrAuthInt4 = [a for a in YrAuthRankDict if YrAuthRankDict[a]>=Thresh4 and YrAuthRankDict[a]<Thresh3] #501-1500
		YrAuthInt5 = [a for a in YrAuthRankDict if YrAuthRankDict[a]>=Thresh5 and YrAuthRankDict[a]<Thresh4] #1501-4000
		#YrAuthInt6 = [a for a in YrAuthRankDict if YrAuthRankDict[a]<Thresh5] #>4000

		print "Getting avg rank for author 1-20 "
		Int1Pap=[p for p in YrPapers if set(PapAuthDict[p]).intersection(YrAuthInt1)!=set()]
		Int1Ref=[]
		for p in Int1Pap:
			refs=PapRefDict[p]
			for r in refs:
				if PapYrDict[r]<=yr:
					Int1Ref+=[r]
		Int1RankSum = sum(YrNoOfPap-YrPapRankDict[r]+1 for r in Int1Ref)
		Int1RankAvg	= Int1RankSum/(len(Int1Ref)*len(Int1Pap))
		Int1YrAvg.append([yr,Int1RankAvg])

		print "Getting avg rank for author 21-100 "
		Int2Pap=[p for p in YrPapers if set(PapAuthDict[p]).intersection(YrAuthInt2)!=set()]
		Int2Ref=[]
		for p in Int2Pap:
			refs=PapRefDict[p]
			for r in refs:
				if PapYrDict[r]<=yr:
					Int2Ref+=[r]
		Int2RankSum = sum(YrNoOfPap-YrPapRankDict[r]+1  for r in Int2Ref)
		Int2RankAvg	= Int2RankSum/(len(Int2Ref)*len(Int2Pap))
		Int2YrAvg.append([yr,Int2RankAvg])

		print "Getting avg rank for author 101-500 "
		Int3Pap=[p for p in YrPapers if set(PapAuthDict[p]).intersection(YrAuthInt3)!=set()]
		Int3Ref=[]
		for p in Int3Pap:
			refs=PapRefDict[p]
			for r in refs:
				if PapYrDict[r]<=yr:
					Int3Ref+=[r]
		Int3RankSum = sum(YrNoOfPap-YrPapRankDict[r]+1 for r in Int3Ref)
		Int3RankAvg	= Int1RankSum/(len(Int3Ref)*len(Int3Pap))
		Int3YrAvg.append([yr,Int3RankAvg])

		print "Getting avg rank for author 501-1500 "
		Int4Pap=[p for p in YrPapers if set(PapAuthDict[p]).intersection(YrAuthInt4)!=set()]
		Int4Ref=[]
		for p in Int4Pap:
			refs=PapRefDict[p]
			for r in refs:
				if PapYrDict[r]<=yr:
					Int4Ref+=[r]
		Int4RankSum = sum(YrNoOfPap-YrPapRankDict[r]+1  for r in Int4Ref)
		Int4RankAvg	= Int1RankSum/(len(Int4Ref)*len(Int4Pap))
		Int4YrAvg.append([yr,Int4RankAvg])


		print "Getting avg rank for author 1501-4000 "
		Int5Pap=[p for p in YrPapers if set(PapAuthDict[p]).intersection(YrAuthInt5)!=set()]
		Int5Ref=[]
		for p in Int5Pap:
			refs=PapRefDict[p]
			for r in refs:
				if PapYrDict[r]<=yr:
					Int5Ref+=[r]
		Int5RankSum = sum(YrNoOfPap-YrPapRankDict[r]+1 for r in Int5Ref)
		Int5RankAvg	= Int1RankSum/(len(Int5Ref)*len(Int5Pap))
		Int5YrAvg.append([yr,Int5RankAvg])

	#In plot better papers and authors have lower index

	Int1YrAvg = sorted(Int1YrAvg,key=itemgetter(0))
	Int2YrAvg = sorted(Int2YrAvg,key=itemgetter(0))
	Int3YrAvg = sorted(Int3YrAvg,key=itemgetter(0))
	Int4YrAvg = sorted(Int4YrAvg,key=itemgetter(0))
	Int5YrAvg = sorted(Int5YrAvg,key=itemgetter(0))
	Int6YrAvg = sorted(Int6YrAvg,key=itemgetter(0))

	p = "/home/siddharth/Desktop/PROJECT/ERP/csmag/graphs/authrefrank/AuthAvgRefRankNormalized1990-2015"
	plt.xlabel("Year")
	plt.ylabel("Avg Ref Rank")
	X=[e[0] for e in Int1YrAvg]
	Y=[e[1] for e in Int1YrAvg]
	plt.plot(X,Y,'ro-',label='1-20')

	X=[e[0] for e in Int2YrAvg]
	Y=[e[1] for e in Int2YrAvg]
	plt.plot(X,Y,'bo-',label='21-100')

	X=[e[0] for e in Int3YrAvg]
	Y=[e[1] for e in Int3YrAvg]
	plt.plot(X,Y,'go-',label='101-500')

	X=[e[0] for e in Int4YrAvg]
	Y=[e[1] for e in Int4YrAvg]
	plt.plot(X,Y,'yo-',label='501-1500')

	X=[e[0] for e in Int5YrAvg]
	Y=[e[1] for e in Int5YrAvg]
	plt.plot(X,Y,'co-',label='1501-4000')

	plt.legend(loc='upper left')
	plt.savefig(p)
	plt.close()

preparedata()
PlotAvgRank()
		
			
