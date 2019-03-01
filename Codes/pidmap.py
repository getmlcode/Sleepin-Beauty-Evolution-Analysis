from collections import defaultdict
ref=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPapYr.txt")

D=defaultdict(list)
D2=defaultdict(list)
a=1
for line in ref:
    line = line.strip()
    line = line.split()
    D[line[0]]=a
    a=a+1
    
ref.close()

cit=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsIntId.txt","w")

for k in D:
    l= k +" "+ str(D[k])+"\n"
    cit.write(l)
cit.close()
