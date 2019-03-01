from collections import defaultdict
ref=open("F:\Acads\IISc ME\project\datasets\citeseer\dates.txt")

D=defaultdict(list)

for line in ref:
    line = line.rstrip()
    if line.strip()==line:
        a=line
    else:
        line = line.lstrip()
        if line.isdigit():
            D[a]=line
ref.close()

cit=open("F:\Acads\IISc ME\project\datasets\citeseer\pap_year.txt","w")
for k in D:
    l= k +" "+ D[k]+"\n"
    cit.write(l)
cit.close()
