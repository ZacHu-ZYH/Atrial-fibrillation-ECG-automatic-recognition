# https://github.com/mondejar/ecg-classification.git

from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists

dir = '../afpdb/'#'mitdb/'
#Create folder
dir_out = '../afpdbCSV/'
if not exists(dir_out):
	mkdir(dir_out)

records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.dat') != -1)]
# print(records)

for r in records:
	# print(r)
	# --> Create Csv files
	command = 'rdsamp -r ' + dir + r[:-4] + ' -c -H -f 0 -t 10 -v>' + dir_out + r[:-4] + '.csv'
	print(command)
	system(command)
	#  --> Create annotation files
	# command_annotations = 'rdann -r ' + dir + r[:-4] +' -f 0 -a atr -v >' + dir_out + r[:-4] + 'annotations.txt'
	# print(command_annotations)
	# system(command_annotations)
   

# system(command_annotations)
records.sort()
print(records)
print(len(records))