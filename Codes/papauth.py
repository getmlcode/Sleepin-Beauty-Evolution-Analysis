
papint={}
authint={}

CsPapIntId=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPapIntId.txt")
CsAuthIntId=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsAuthIntId.txt")
CsPapAuth=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/CsPapAuth.txt")

CsPidAid=open("/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt","w")


for line in CsPapIntId:
    line = line.strip()
    line = line.split()
    papint[line[0]] = line[1]
print "Pap-Int dictionary created\n"    
CsPapIntId.close()

for line in CsAuthIntId:
    line = line.strip()
    line = line.split('\t')
    authint[line[0]] = line[1]
print "Auth-Int dictionary created\n" 
   
CsAuthIntId.close()


print "Creating CsPidAid.txt file\n" 
for line in CsPapAuth:
    line = line.strip()
    line = line.split('\t')
    l = str(papint[line[0]])+"\t"+str(authint[line[1]])+"\n"
    CsPidAid.write(l)
print "CsPidAid.txt file Created\n" 
CsPidAid.close()
