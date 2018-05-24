import json
from random import shuffle

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data


miami_data = open_json('miami-data.json')
data = miami_data ## add twitter-data.json


train = []
val = []
test = []
code_switched = []
non_code_switched = []

for d in data:
	eng = False
	spa = False

	for token in d:
		lang = token[1]
		if lang == 'spa' or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa':
			spa = True
		if lang == 'eng'or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa':
			eng = True

	if eng and spa: code_switched.append(d)
	else: non_code_switched.append(d)

shuffle(code_switched)
shuffle(non_code_switched)

train = code_switched[0:-2000] + non_code_switched
val_test = code_switched[-2000:]
val = val_test[:1000]
test = val_test[1000:]

print('Train size: ', len(train))
print('Val size: ', len(val))
print('Test size: ', len(test))

with open('train.json', 'w') as outfile:
    json.dump(train, outfile)

with open('val.json', 'w') as outfile:
    json.dump(val, outfile)

with open('test.json', 'w') as outfile:
    json.dump(test, outfile)

