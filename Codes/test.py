import numpy as np
from collections import defaultdict

'''
# Save
dictionary = {1:'d'}
np.save('my_dict.npy', dictionary) 

# Load
read_dictionary = np.load('my_dict.npy').item()
print(read_dictionary[1]) # displays "world"
'''


print "loading dictionaries"
PapAuthDict = np.load('PapAuthDict.npy').item()
AuthPapDict = np.load('AuthPapDict.npy').item()

PapRefDict = np.load('PapRefDict.npy').item()
PapCitDict = np.load('PapCitDict.npy').item()

PapYrDict = np.load('PapYrDict.npy').item()
YrPapDict = np.load('YrPapDict.npy').item()
print "Dictionaries Loaded"



'''
def preparedata():
	print "\nLoading Files"
	PapAuth=np.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidAid.txt')
	PapYr=np.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidYr.txt')
	PaperPaper=np.loadtxt('/home/siddharth/Desktop/PROJECT/Datasets/CSMAG/INT/CsPidRefPid.txt')

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
		YrPapDict[int(row[1])]+=[int(row[0])]

print "\nCreating Dictionaries"
preparedata()

print "\nSaving Dictionaries"

np.save('PapAuthDict.npy', PapAuthDict)
np.save('AuthPapDict.npy', AuthPapDict)

np.save('PapRefDict.npy', PapRefDict)
np.save('PapCitDict.npy', PapCitDict)

np.save('PapYrDict.npy', PapYrDict)
np.save('YrPapDict.npy', YrPapDict)

print "\nDictionaries Saved"
'''
