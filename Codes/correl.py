from __future__ import division
import numpy



#f=open("/home/siddharth/Desktop/PROJECT/ERP/csmag/WakeupPoint/CsCitRateRankNew.txt","r")
f=open("/home/siddharth/Desktop/PROJECT/ERP/csmag/WakeupPoint/CsMyWakePeriodNew_11.txt","r")
cit=[]
rate=[]
rank=[]
wd=[]
for l in f:
	l=l.strip().split()
	if len(l) > 0:
		#print l
		wd.append(int(l[3]))
		cit.append(int(l[4]))
		rate.append(float(l[5]))
		rank.append(int(l[6]))

rank = [1/r for r in rank]

'''
print cit
print rate
print rank
'''
print "cit rank correlation :"
print numpy.corrcoef(cit,rank)[0, 1]
print "rate rank correlation :"
print numpy.corrcoef(rate,rank)[0, 1]
print "wakeup duration rank correlation :"
print numpy.corrcoef(wd,rank)[0, 1]







'''



