from __future__ import division
from collections import defaultdict
from operator import itemgetter
from collections import Counter
import numpy
import scipy.io
import random
import math


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
YrPapRankDict={}
YrAuthRankDict={}

YrPap=[]
fcp=[]


YrNoOfAuth=0
YrNoOfPap=0
YrLeadRank=0
YrFolRank=0
RefRank=0
NoOfPap=0
NoOfInfAuth=0

#-----------------------------Functions---------------------------------
#1. To load necessary files and create required dictionaries
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
		PapCitDict[int(row[1])]+=[int(row[0])]

	print "\nCreating year-paper and paper-year Dictionaries"
	for row in PapYr:
		PapYrDict[int(row[0])]=int(row[1])
		YrPapDict[int(row[1])]+=[int(row[0])]

	print "Creating papers upto year dictionary"
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
	print "Authors upto year dictionary created"




preparedata()
pra=[1963]+range(1972,2016)

for yr in pra:
	if yr > 2000:
		print "writing paper and auth rank files for year %d"%(yr)
		PapRankDict={}
		path = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseRanks/'+str(yr)+'Rank.txt'
		YrRank = open(path,"r")
		for line in YrRank:
			line = line.strip().split('\t')
			PapRankDict[int(line[0])]=float(line[1])

		YrPap = UptoYrPapDict[yr]
		YrPap.remove(1)
		YrAuth = UptoYrAuthDict[yr]	
	
		YrPapRankDict={p:PapRankDict[p] for p in YrPap}
		PapScoreList = [ [x,YrPapRankDict[x]] for x in YrPapRankDict.keys()]
		PapScoreList = sorted(PapScoreList, key=itemgetter(1))

		path2 = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWisePapRanks/'+str(yr)+'PapRank.txt'
		paprank = open(path2,"w") # higher number is better
		l=""
		for p in range(len(PapScoreList)):
			l+=str(PapScoreList[p][0])+"\t"+str(p+1)+"\n"
		paprank.write(l)

		UptoYrAuthPapDict=defaultdict(list)
		for a in YrAuth:
			yrAuthPap = [p for p in AuthPapDict[a] if PapYrDict[p] <=yr ]
			#UptoYrAuthPapDict[a]+=list(set(AuthPapDict[a]).intersection(set(UptoYrPapDict[yr])))
			UptoYrAuthPapDict[a] = yrAuthPap
	
		AuthRankList=[]
		for auth in UptoYrAuthPapDict.keys():
			authpap = UptoYrAuthPapDict[auth]
			ARscore = 0
			for p in authpap:
				if int(p) != 1:
					ARscore = ARscore + PapRankDict[int(p)]
			AuthRankList.append([auth,ARscore])

		AuthRankList=sorted(AuthRankList, key=itemgetter(1))

		Arank = 0
		path2 = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseAuthRanks/'+str(yr)+'AuthRank.txt'
		authrank = open(path2,"w") # higher number is better
		line=""
		for l in AuthRankList:
			Arank = Arank+1
			line+=str(l[0])+"\t"+str(Arank)+"\n"
		authrank.write(line)

		YrRank.close()
		paprank.close()
		authrank.close()

