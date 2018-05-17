import os

folder = 'miami-data'
files = os.listdir(folder)

for file_name in files:
	if not file_name.endswith('.cha'): continue
	with open(folder + '/' + file_name) as file:
		for line in file: 
			print(line)
