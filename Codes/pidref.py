from collections import defaultdict


D=defaultdict(int)
D2=defaultdict(list)

ref=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsIntId.txt")

for line in ref:
    line = line.strip()
    line = line.split()
    D[line[0]] = line[1]
    
ref.close()

ref=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPapRef.txt")
cit=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPidRefPid.txt","w")

for line in ref:
    line = line.strip()
    line = line.split()
    l = str(D[line[0]])+"\t"+str(D[line[1]])+"\n"
    cit.write(l)

ref.close()   
cit.close()
