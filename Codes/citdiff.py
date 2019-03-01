from __future__ import division
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import numpy
import scipy.io
import random
import math


PapAuthDict=defaultdict(list)
AuthPapDict=defaultdict(list)
PapRefDict=defaultdict(list)
PapCitDict=defaultdict(list)
PapYrDict=defaultdict(list)
FracDiffDict={}


def preparedata():
	print "\nLoading Files"
	PapAuth=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
	PapYr=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidYr.txt')
	PaperPaper=numpy.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidRefPid.txt')

	print "\nCreating paper-aurthor and author-paper Dictionaries"
	for row in PapAuth:
		PapAuthDict[int(row[0])]+=[int(row[1])]
		AuthPapDict[int(row[1])]+=[int(row[0])]

	print "\nCreating reference and citations Dictionaries"
	for row in PaperPaper:
		PapRefDict[int(row[0])]+=[int(row[1])]
		PapCitDict[int(row[1])]+=[int(row[0])]

	print "\nCreating year-paper and paper-year Dictionaries"
	for row in PapYr:
		PapYrDict[int(row[0])]+=[int(row[1])]

preparedata()
authors = AuthPapDict.keys()
n = len(authors)
option = 1
while option != 0:
	A1 = int(raw_input("Enter Author Id : "))
	A1Ref = [PapRefDict[p] for p in AuthPapDict[A1]]
	A1Ref = reduce(lambda x,y: x+y,A1Ref)
	A1TotRef = len(A1Ref)
	AuthCitDiffList = []
	for A2 in authors:
		if A1 != A2:
			FracDiffDict[A1]=defaultdict(float)
			FracA1toA2=0
			FracA2toA1=0
			A2Ref = [PapRefDict[p] for p in AuthPapDict[A2]]
			A2Ref = reduce(lambda x,y: x+y,A2Ref)
			A2TotRef = len(A2Ref)
			A1toA2 = sum(A2 in PapAuthDict[r] for r in A1Ref)
			A2toA1 = sum(A1 in PapAuthDict[r] for r in A2Ref)

			if A1TotRef!=0:
				FracA1toA2 = A1toA2/A1TotRef
			if A2TotRef!=0:
				FracA2toA1 = A2toA1/A2TotRef

			FracDiffA1A2 = FracA1toA2 - FracA2toA1
			AuthCitDiffList.append(FracDiffA1A2)

	#print AuthCitDiffList
	FreqDict = Counter(AuthCitDiffList)
	keys = FreqDict.keys()
	FreqList=[]
	for k in keys:
		FreqList.append([ k,FreqDict[k] ])
	
	FreqList=sorted(FreqList,key=itemgetter(0))
	
	x=[]
	y=[]	
	for l in FreqList:
		x.append(l[0])
		y.append(l[1])
	print "\n difference :\n "
	print x
	print "\nFreq :\n"
	print y
	plt.plot(x,y,'ro-')
	plt.show()
	plt.close()

	option = int(raw_input("Enter 0 to stop : "))
	
