from collections import defaultdict
ref=open("F:\Acads\IISc ME\project\datasets\citeseer\my.txt")

D=defaultdict(list)

for line in ref:
    line = line.rstrip()
    if line.strip()==line:
        a=line
    else:
        line = line.lstrip()
        D[line]=a
ref.close()

cit=open("F:\Acads\IISc ME\project\datasets\citeseer\cit2.txt","w")
for k in D:
    l= k +" "+ D[k]+"\n"
    cit.write(l)
cit.close()
