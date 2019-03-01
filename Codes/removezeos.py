from collections import defaultdict

ref=open("F:\Acads\IISc ME\project\datasets\citeseer\pap_ref_pap.txt")
out=open("F:\Acads\IISc ME\project\datasets\citeseer\pap_ref_pap2.txt","w")


for line in ref:
    line = line.strip()
    line = line.split()
    a = int(line[0])
    b = int(line[1])
    if b!=0 and a!=0:
        l = str(a)+" "+str(b)+"\n"
        out.write(l)

ref.close()
out.close()
