from __future__ import division
from collections import defaultdict
from operator import itemgetter
from collections import Counter
import numpy
import scipy.io
import random
import math

#---------------------------Global Variables----------------------------
SB=set([29348,344538,412928,523387,235719,466535,34678,406001,520119,520628,478023,466547,66242,347464,104994,236251,89195,165321,499215,466838])

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
YrDone={}

YrPap=[]
fcp=[]
pra=range(1972,2016)+[1963]

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
	

	print "\nCreating author reference history Dictionary"
	for a in AuthPapDict:
		paps = AuthPapDict[a]
		AuthRefHistDict[a]=defaultdict(list)
		for p in paps:
			papref = PapRefDict[p]
			for r in papref:			
				AuthRefHistDict[a][r]+=[PapYrDict[p]]
	print "\nAuthor reference history Dictionary created"


class web:
    def __init__(self,n):
        self.size = n
        self.in_links = {}
        self.number_out_links = {}
        self.dangling_pages = {}
        for j in xrange(n):
            self.in_links[j] = []
            self.number_out_links[j] = 0
            self.dangling_pages[j] = True

def step(g,p,s=0.85):
    '''Performs a single step in the PageRank computation,
    with web g and parameter s.  Applies the corresponding M
    matrix to the vector p, and returns the resulting
    vector.'''
    n = g.size
    v = numpy.matrix(numpy.zeros((n,1)))
    inner_product = sum([p[j] for j in g.dangling_pages.keys()])
    for j in xrange(n):
        if j % 10000 == 0:
            print j,
        v[j] = s*sum([p[k]/g.number_out_links[k]
                      for k in g.in_links[j]])+s*inner_product/n+(1-s)/n
    # We rescale the return vector, so it remains a
    # probability distribution even with floating point
    # roundoff.
    return v/numpy.sum(v)

def pagerank(g,s=0.85,tolerance=0.000001):
    '''Returns the PageRank vector for the web g and
    parameter s, where the criterion for convergence is that
    we stop when M^(j+1)P-M^jP has length less than
    tolerance, in l1 norm.'''
    n = g.size
    p = numpy.matrix(numpy.ones((n,1)))/n
    iteration = 1
    change = 2
    while change > tolerance:
        print "Iteration: %s" % iteration
        new_p = step(g,p,s)
        change = numpy.sum(numpy.abs(p-new_p))
        print "\nChange in l1 norm: %s" % change
        p = new_p
        iteration += 1
    return p

def GetNewRank(YrPaper_Paper,yr):
    print 'Constructing citation network...'
    #citations = open('/home/anubhav/Desktop/Datasets/PubMed_subset-master/paper_paper.txt', 'r')
    paprank={}
    g = web(528719)
    for line in YrPaper_Paper:
        # Here j is citing paper and k is cited paper
        j, k = map(int, line)
        g.number_out_links[j-1] += 1
        g.in_links[k-1].append(j-1)
        g.dangling_pages[j-1] = False
    print '\nComputing PageRank...\n'
    pr = pagerank(g, 0.85, 0.000001)
    rank = numpy.array(map(float, pr))
    idx = rank.argsort()

    ans=""
    path = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseRanks/'+str(yr)+'Rank.txt'
    output = open(path, "w")
    for i in range(1, 528719):
        paprank[int(idx[-i]+1)]=float(rank[idx[-i]])
        ans += str(idx[-i]+1) + "\t" + str(rank[idx[-i]]) + '\n'
    output.write(ans)
    output.close() 
    return paprank

#2. To get frequently cited papers upto a given year
def GetFCP(PapCitDict,yr):
	PapCitCount={}
	CountList=[]
	for p in PapCitDict:
		if PapYrDict[p] <= yr:
			cit = PapCitDict[p]
			CitYrList = [PapYrDict[x] for x in cit]
			YrCitCount = sum(el<=yr for el in CitYrList)
			PapCitCount[p] = YrCitCount
			CountList.append(YrCitCount)
	
	CutOffCit = math.ceil(0.7*max(CountList)) 
	fcp = [pap for pap in PapCitCount if PapCitCount[p] > CutOffCit]
	return fcp
	
#3. To calculate rank of authors and papers for a given year
def GetRank(yr,lead,fol):
	global YrAuthRankDict
	global YrPapRankDict
	global YrNoOfAuth
	global YrNoOfPap
	global YrPap
	global YrPapRankDict
	global YrAuthRankDict
	global YrDone
	global pra

	PapRankDict={}
	YrPap = UptoYrPapDict[yr]
	YrAuth = UptoYrAuthDict[yr]
	YrNoOfAuth = len(YrPap)
	YrNoOfPap = len(YrAuth)

	if yr in pra:
		#print "Reading Page Rank from already present file"
		pappath = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWisePapRanks/'+str(yr)+'PapRank.txt'
		authpath = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseAuthRanks/'+str(yr)+'AuthRank.txt'

		YrPapRankDict[yr]=defaultdict(int)
		YrRank = open(pappath,"r")
		for line in YrRank:
			line = line.strip().split('\t')
			YrPapRankDict[yr][int(line[0])]=int(line[1])

		YrAuthRankDict[yr]=defaultdict(int)
		YrAuRank = open(authpath,"r")
		for line in YrAuRank:
			line = line.strip().split('\t')
			YrAuthRankDict[yr][int(line[0])]=int(line[1])

		YrDone[yr]=yr
		YrRank.close()
		YrAuRank.close()

	else:

		print "Calculating Page Rank for year %d"%(yr)
		BeforeYrPap=UptoYrPapDict[yr]
		for p in YrPap:
			ref = set(paper_referencesDict[p])
			ref = ref.intersection(BeforeYrPap)
			for r in ref:
				YrPapREf.append([p,r])
		PapRankDict = GetNewRank(YrPapREf,yr)
		pra.append(yr)

		YrPap.remove(1)

		YrrPapRankDict={p:PapRankDict[p] for p in YrPap}
		PapScoreList = [ [x,YrrPapRankDict[x]] for x in YrrPapRankDict.keys()]
		PapScoreList = sorted(PapScoreList, key=itemgetter(1))
		path2 = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWisePapRanks/'+str(yr)+'PapRank.txt'
		paprank = open(path2,"w") # higher number is better
		l=""
		for p in range(len(PapScoreList)):
			l+=str(PapScoreList[p][0])+"\t"+str(p+1)+"\n"
			YrPapRankDict[yr][PapScoreList[p][0]] = int(p+1)
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
			YrAuthRankDict[yr][l[0]] = int(Arank)
		authrank.write(line)

		YrRank.close()
		paprank.close()
		authrank.close()
		YrDone[yr]=yr

	'''
	if lead in YrAuth:
		print "lead is present"
	else :
		print "lead not present"
	if fol in YrAuth:
		print "fol is present"
	else :
		print "fol not present"
	'''
	'''if lead in YrAuthRankDict.keys():
		print "lead is present in rank dict"
	else :
		print "lead not present in rank dict"

	if fol in YrAuthRankDict.keys():
		print "fol is present in rank dict"
	else :
		print "fol not present in rank dict"'''
	
	#print "Rank assignment for year %d completed" %(yr)

#4 To Find Influential Status of authors
def GetInflStatus(YrLeadRank,YrFolRank):
	global NoOfInfAuth
	NoOfInfAuth = YrNoOfAuth-.99*YrNoOfAuth
	Threshold = YrNoOfAuth - NoOfInfAuth + 1

	if YrLeadRank >= Threshold and YrFolRank < Threshold:
		return 10
	
	if YrLeadRank < Threshold and YrFolRank < Threshold:
		return 00

	if YrLeadRank < Threshold and YrFolRank >= Threshold:
		return 01

	if YrLeadRank >= Threshold and YrFolRank >= Threshold:
		return 11

#5 To get Citation Chasing Probability(simple)
def GetCitChaseProbS(lead,fol):
	CitChaseProb = 0

	yearPap = set(YrPap)-set(fcp)

	leadPap   = set(AuthPapDict[lead])
	folPap    = set(AuthPapDict[fol])

	YrleadPap = list(yearPap.intersection(folPap))
	YrfolPap  = list(yearPap.intersection(folPap))

	folRef    = [list(set(PapRefDict[p]).intersection(yearPap)) for p in YrfolPap]
	folRef    = reduce(lambda x,y: x+y,folRef)

	leadRef   = [list(set(PapRefDict[p]).intersection(yearPap)) for p in YrleadPap]
	leadRef   = reduce(lambda x,y: x+y,leadRef)
	commRef   = set(folRef).intersection(set(leadRef))

	folTotRef 	 = len(folRef)
	leadTotRef 	 = len(leadRef)
	TotCommRef = len(commRef)
	TotFtoLRef  = sum(lead in PapAuthDict[r] for r in folRef)
	

	if folTotRef!=0 and leadTotRef!=0 and folTotRef!=0: 
		FtoLProb = TotFtoLRef/folTotRef
		leadCommProb = TotCommRef/leadTotRef
		folCommProb = TotCommRef/folTotRef

		CitChaseProb = FtoLProb*leadCommProb

	return CitChaseProb
	
#6 To get Citation Chasing Probability(complicated)
def GetCitChaseProbC(lead,fol,year):
	CitChaseProb = 0
	yearPap = set(YrPap)-set(fcp)

	leadPap   = AuthPapDict[lead]
	folPap    = AuthPapDict[fol]

	YrleadPap = [p for p in leadPap if PapYrDict[p]<=yr]
	YrfolPap  = [p for p in folPap if PapYrDict[p]<=yr]

	YrleadRef = []
	for p in YrleadPap:
		YrleadRef = YrleadRef+PapRefDict[p]
	leadTotRef 	 = len(YrleadRef)

	YrfolRef = []
	for p in YrfolPap:
		YrfolRef = YrfolRef+PapRefDict[p]
	folTotRef 	 = len(YrfolRef)
	
	FtoLRef = [r for r in YrfolRef if lead in PapAuthDict[r]]
	TotFtoLRef  = len(FtoLRef)	

	cr  = set(YrleadRef).intersection(set(YrfolRef))
	lrd = Counter(YrleadRef)
	frd = Counter(YrfolRef)
	crd = {k:(lrd[k]<crd[k])*lrd[k]+(lrd[k]>=crd[k])*crd[k] for k in cr}
	commRef=[]
	for k in crd:
		commRef.extend([k]*crd[k])
	TotCommRef = len(commRef)	

	#TotFtoLRef  = sum(lead in PapAuthDict[r] for r in YrfolRef)

	chaseRef = sum( (set(PapRefDict[p).intersection(commRef))!=set() for p in FtoLRef )

	if folTotRef != 0 and chaseRef != 0:
		CitChaseProb = chaseRef/folTotRef
	
	return CitChaseProb


#7 To calculate Following Coefficient between authors
def FollowingCoeff(fol,lead):
	FolLeadCoeffList=[]
	LPap = AuthPapDict[lead]
	FPap = AuthPapDict[fol]
	F_FirstPub=2017
	F_LastPub=1900
	L_FirstPub=2017
	L_LastPub=1900
	FolRef=[]
	LeadRef=[]
	ExpRatioFolScore = 0.0
	RatioFolScore = 0.0

	for p in FPap:
		FolRef += PapRefDict[p]
		if F_FirstPub > PapYrDict[p]:
			F_FirstPub = PapYrDict[p]
		if F_LastPub < PapYrDict[p]:
			F_FirstPub = PapYrDict[p]
	for p in LPap:
		LeadRef += PapRefDict[p]
		if L_FirstPub > PapYrDict[p]:
			L_FirstPub = PapYrDict[p]
		if L_LastPub < PapYrDict[p]:
			L_FirstPub = PapYrDict[p]

	FolRef = set(FolRef)
	LeadRef = set(LeadRef)-set([1])
	CommRef = list(FolRef.intersection(LeadRef)-SB) #Excluding Sleeping Beauty Papers From Reference Chasing Behavior
	
	
	if len(CommRef) > 0:
		#print "Common references between author %d and author %d"%(fol,lead)
		#print CommRef
		for r in CommRef:
			LeadRefYr = list(set(sorted(AuthRefHistDict[lead][r])))
			FolRefYr =  list(set(sorted(AuthRefHistDict[fol][r])))
			'''print "lead %d ref yr : " %(lead)
			print LeadRefYr
			print "Fol %d ref yr : " %(fol)
			print FolRefYr'''
			i=0
			j=0
			while i<len(LeadRefYr) and j<len(FolRefYr):
				if LeadRefYr[i] != FolRefYr[j]:
					if FolRefYr[j] <= LeadRefYr[i]+5 and FolRefYr[j] > LeadRefYr[i]:
						NextRefYR = LeadRefYr[i]+6
						if F_FirstPub >= FolRefYr[j] or L_FirstPub >= FolRefYr[j] or PapYrDict[r]>=FolRefYr[j]:
							yr = FolRefYr[j]
						else :
							yr = FolRefYr[j]-1
						#print "Getting frequently cited papers upto year %d"%(yr)
						fcp = GetFCP(PapCitDict,yr)
						#print "Frequently cited papers calculated"
						if r not in fcp:
							if yr in YrDone.keys():
								YrLeadRank = YrAuthRankDict[yr][lead]
								YrFolRank  = YrAuthRankDict[yr][fol]
								YrRefRank =  YrPapRankDict[yr][r]
							else :
								GetRank(yr,lead,fol) #rank is calculated for year previous to year of referrence(higher value for higher rank author)
								YrLeadRank = YrAuthRankDict[yr][lead]
								YrFolRank  = YrAuthRankDict[yr][fol]
								YrRefRank =  YrPapRankDict[yr][r]
							RefInvPerc = 100*(YrNoOfPap - YrRefRank)/YrNoOfPap
							#print "Getting influential status of author %d and author %d in year %d"%(fol,lead,yr)
							status = GetInflStatus(YrLeadRank,YrFolRank)
							if status == 10:
								#print "Getting citation chasing probability"
								CitChaseS=GetCitChaseProbS(lead,fol)
								CitChaseC=GetCitChaseProbC(lead,fol)
								if j == 0:
									FolRefGap = FolRefYr[j]-F_FirstPub 
									FolLeadGap = FolRefYr[j] - LeadRefYr[i]
									ExpRatioFolScore += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseS
									RatioFolScore += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseS
									ExpRatioFolScoreC += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseC
									RatioFolScoreC += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseC
								else :
									FolRefGap = FolRefYr[j]-FolRefYr[j-1]
									FolLeadGap = FolRefYr[j] - LeadRefYr[i]
									ExpRatioFolScore += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseS
									RatioFolScore += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseS
									ExpRatioFolScoreC += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseC
									RatioFolScoreC += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*CitChaseC
								while LeadRefYr[i] < NextRefYR:
									i+=1
									if i >= len(LeadRefYr):
										break
								while FolRefYr[j] < NextRefYR:
									j+=1
									if j >= len(FolRefYr):
										break

							elif status == 11:
								#print "Getting citation chasing probability"
								CitChaseS=GetCitChaseProbS(lead,fol)
								ego = math.exp(-1*NoOfInfAuth/abs(YrLeadRank-YrRefRank))
								CitChaseC=GetCitChaseProbC(lead,fol)
								if j == 0:
									FolRefGap = FolRefYr[j]-F_FirstPub
									FolLeadGap = FolRefYr[j] - LeadRefYr[i]
									ExpRatioFolScore += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseS
									RatioFolScore += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseS
									ExpRatioFolScoreC += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseC
									RatioFolScoreC += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseC
								else :
									FolRefGap = FolRefYr[j]-FolRefYr[j-1]
									FolLeadGap = FolRefYr[j] - LeadRefYr[i]
									ExpRatioFolScore += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseS
									RatioFolScore += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseS
									ExpRatioFolScoreC += RefInvPerc*math.exp(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseC
									RatioFolScoreC += RefInvPerc*(FolRefGap/FolLeadGap)*(YrLeadRank/YrNoOfAuth)*ego*CitChaseC	
								while LeadRefYr[i] < NextRefYR:
									i+=1
									if i >= len(LeadRefYr):
										break
								while FolRefYr[j] < NextRefYR:
									j+=1
									if j >= len(FolRefYr):
										break
							else :
								while LeadRefYr[i] < NextRefYR:
									i+=1
									if i >= len(LeadRefYr):
										break
								while FolRefYr[j] < NextRefYR:
									j+=1
									if j >= len(FolRefYr):
										break
								
					elif FolRefYr[j] > LeadRefYr[i]+5:
						i+=1
					elif FolRefYr[j] < LeadRefYr[i]:
						j+=1
				else:
					i+=1
					j+=1

		ExpRatioFolScore=int(ExpRatioFolScore)+round(ExpRatioFolScore-int(ExpRatioFolScore),5)
		RatioFolScore=int(RatioFolScore)+round(RatioFolScore-int(RatioFolScore),5)
		FolLeadCoeffList = [ExpRatioFolScore,RatioFolScore]
		#print "Following coefficient of author %d w.r.t author %d calculated"%(fol,lead)
		return FolLeadCoeffList
	return 0


#-------------------------------------Main code---------------------------------------

preparedata() # Function To Prepare necessary dictioanries
Authors = AuthPapDict.keys()
option = 1

while option!=0:
	fol=int(raw_input("Enter Following Author ID :"))
	print "Calculating following coefficient of author %d w.r.t other authors "%(fol)
	for lead in Authors:
		#print "Calculating following coefficient of author %d w.r.t author %d"%(fol,lead)
		if lead != fol:
			FolLeadCoefDict[fol]=defaultdict(list)
			FolCoef = FollowingCoeff(fol,lead)
			if FolCoef != 0:
				FolLeadCoefDict[fol][lead] = FolCoef
	print "Calculation of Following Coefficient of Author %d Completed"%(fol)
	option=int(raw_input("Enter 0 To Exit :"))

op=1
while op!=0:
	ld=int(raw_input("Enter Lead ID : "))
	if ld in FolLeadCoefDict[fol].keys():
		print "Following Coefficient of %d w.r.t %d : %f" %(fol,ld,FolLeadCoefDict[fol][lead])
	else:
		print "No author with such ID exists"
	op=int(raw_input("Enter 0 to exit : "))
