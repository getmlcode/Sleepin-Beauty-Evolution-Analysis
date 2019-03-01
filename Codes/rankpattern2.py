from collections import defaultdict
from operator import itemgetter
import numpy
import scipy.io
import time
import random


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

def GetRank(YrPaper_Paper,yr):
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
#---------------------------------------Loading Necessary Files-------------------------
print "\nLoading Files"
PapCitPattern=scipy.io.loadmat('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPapCitPattern.mat')
PapCitPattern=PapCitPattern['CsPapCitPattern']
p_year=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidYr.txt')
paper_paper=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidRefPid.txt')
paper_author=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
print "\nFiles Loaded"
#------------------------------------Creating Necessary Dictionaries--------------------
print "\nCreating Necessary Dictionaries"
PapCitPatDict = defaultdict(lambda : defaultdict(int))
paper_referencesDict = defaultdict(list)
paper_citationsDict = defaultdict(list)
paper_yearDict = defaultdict(list)
year_paperDict = defaultdict(list)
paper_authorDict = defaultdict(list)
author_paperDict = defaultdict(list)
CitYrsDict = defaultdict(list)
print "\nDictionaries Created"

d=defaultdict(int)
d={}
i=1
print "\nCreating Citation Pattern and Citation Years Dictionaries"
for row in PapCitPattern:
    if i !=row[0]:
        d={}
        i=row[0]
    d[row[1]]=row[2]
    PapCitPatDict[row[0]]=d
    CitYrsDict[row[0]]=d.keys()
print "\nCitation Pattern Dictionary and Cit Years Dict Created"

print "\nCreating reference and citations Dictionaries"
for row in paper_paper:
    paper_referencesDict[int(row[0])]+=[int(row[1])]
    paper_citationsDict[int(row[1])]+=[int(row[0])]
print "\nReference and citations Dictionaries Created"

print "\nCreating paper-aurthor and author-paper Dictionaries"
for row in paper_author:
    paper_authorDict[int(row[0])]+=[int(row[1])]
    author_paperDict[int(row[1])]+=[int(row[0])]

print "\nCreating year-paper and paper-year Dictionaries"
for row in p_year:
    paper_yearDict[int(row[0])]+=[int(row[1])]
    year_paperDict[int(row[1])]+=[int(row[0])]

#------------------------------------Main Program Starts Here--------------------


t='1'

done=[1963,1972,1973,1974,1980,1981,1983,1984,1985,1987,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
pidlist=[29348,344538,412928,523387,235719,466535,34678,406001,520119,520628,478023,466547,66242,347464,104994,236251,89195,165321,499215,466838]

for pid in pidlist:
    print "\n --------------------------------STARTING RANKPATTERN ANALYSIS FOR PAPER %d-------------------------------------"%(pid)
    print "\n Paper %d Received Citations In Following Years:"%(pid)
    print CitYrsDict[pid]

    pidHighRank=[]
    pidMidRank=[]
    for yr in CitYrsDict[pid]:
        
		  # citing papers and authors in particular year
        YrPapers = set(year_paperDict[yr])
        Citers = set(paper_citationsDict[pid])
        YrCiters = list(YrPapers.intersection(Citers))
        YrAuthors=[]
	
        for p in YrCiters:
	    	  YrAuthors.append(paper_authorDict[p])
        YrAuthors=reduce(lambda x,y: x+y,YrAuthors)
        YrAuthors = list(set(YrAuthors))
		
        print "\nCiting Papers in year %d for paper %d :" %(yr,pid)
        print YrCiters
        print "\nCiting Authors  in year %d for paper %d :" %(yr,pid)
        print YrAuthors
		
			  # YrPaper_paper dict
        print "\nExtracting Refferences upto Year %d"%(yr)
        BeforeYrPap=[]
        for y in year_paperDict:
			  if y <= yr:
			      for p in year_paperDict[y]:
			          BeforeYrPap.append(p)

        YrPaper_Paper=[]
        BeforeYrPap=set(BeforeYrPap)
        BeforeYrAuth=defaultdict(list) #papers by auth upto year Yr
        for p in BeforeYrPap:
			  ref = set(paper_referencesDict[p])
			  ref = ref.intersection(BeforeYrPap)
			  for a in paper_authorDict[p]:
			      BeforeYrAuth[a]+=[p]
			  for r in ref:
			      YrPaper_Paper.append([p,r])
        print "\nReferrences upto year %d Extracted :)"%(yr)
		
        print "\nCalculating Rank upto year %d"%(yr)
        YrPapRankDict={}
        if yr in done:
            path = '/home/siddharth/Desktop/PROJECT/ERP/csmag/YearWiseRanks/'+str(yr)+'Rank.txt'
            YrRank = open(path,"r")
            print "\nReading Ranks From Already Present File"
            for line in YrRank:
                    line = line.strip().split('\t')
                    YrPapRankDict[int(line[0])]=float(line[1])
        else:
            YrPapRankDict = GetRank(YrPaper_Paper,yr)
            done.append(yr)

        AuthRankList=[]
        AuthRankDict={}
        AuthRankScore={}
        YrAuthRankList=[]

        for auth in BeforeYrAuth:
            authpap = BeforeYrAuth[auth]
            ARscore = 0
            for p in authpap:
                if int(p) != 1:
                   ARscore = ARscore + YrPapRankDict[int(p)]
            AuthRankList.append([auth,ARscore])
            AuthRankScore[auth]=ARscore

        AuthRankList=sorted(AuthRankList, key=itemgetter(1), reverse=True)

        Arank = 0
        for l in AuthRankList:
            Arank = Arank+1
            AuthRankDict[l[0]] = Arank

        for auth in YrAuthors:
            YrAuthRankList.append([auth,AuthRankDict[auth]])

        YrAuthRankList=sorted(YrAuthRankList, key=itemgetter(1))

        pidHighRank.append([yr,YrAuthRankList[0][0],YrAuthRankList[0][1]])
        mid = len(YrAuthRankList)/2
        pidMidRank.append([yr,YrAuthRankList[mid][0],YrAuthRankList[mid][1]])
    
#-----------------Writing RankPattern Files----------------------------------------------------
    l=""
    pidHighRank=sorted(pidHighRank, key=itemgetter(0), reverse=True)
    for elm in pidHighRank:
        l+=str(elm[0])+'\t'+str(elm[1])+'\t'+str(elm[2])+'\n'
    path = '/home/siddharth/Desktop/PROJECT/ERP/csmag/RankPattern/'+str(pid)+'HighRank.txt'
    hrf = open(path,"w")
    hrf.write(l)
    hrf.close()

    l=""
    pidMidRank=sorted(pidMidRank, key=itemgetter(0), reverse=True)
    for elm in pidMidRank:
        l+=str(elm[0])+'\t'+str(elm[1])+'\t'+str(elm[2])+'\n'
    path = '/home/siddharth/Desktop/PROJECT/ERP/csmag/RankPattern/'+str(pid)+'MidRank.txt'
    mrf = open(path,"w")
    mrf.write(l)
    mrf.close()
    print "\n----------------------------------RANKPATTERN ANALYSIS FOR PAPER %d COMPLETED-----------------------------------"%(pid)
