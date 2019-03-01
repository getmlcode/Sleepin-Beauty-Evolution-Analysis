import os

directory='/home/siddharth/Desktop/PROJECT/ERP/csmag/FolPattern/412928'
if not os.path.exists(directory):
    os.makedirs(directory)
else :
	print "directory already exists"
