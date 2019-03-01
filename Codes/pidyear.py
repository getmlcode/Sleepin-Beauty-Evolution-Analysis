from collections import defaultdict


D=defaultdict(int)


ref=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsIntId.txt")

for line in ref:
    line = line.strip()
    line = line.split()
    D[line[0]] = line[1]
    
ref.close()

ref=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPapYr.txt")
cit=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPidYr.txt","w")

for line in ref:
       
    line = line.strip()
    line = line.split()
    a=line[0]
    b=line[1]
    
    l = str(D[a])+"\t"+str(b)+"\n"
    cit.write(l)

ref.close()   
cit.close()
