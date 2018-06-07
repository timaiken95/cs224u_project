import json
from random import shuffle

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data

# miami_data = open_json('miami-data.json')
# twitter_data = open_json('twitter-lower.json')
# data = miami_data + twitter_data

data = open_json('all-data.json')


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
		if lang == 'spa' or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa' or lang == 'mixed':
			spa = True
		if lang == 'eng'or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa'or lang == 'mixed':
			eng = True

	if eng and spa: code_switched.append(d)
	else: non_code_switched.append(d)

shuffle(code_switched)
shuffle(non_code_switched)

print('Number of code-switched utterances: ', len(code_switched))
print('Number of non-code-switched utterances: ', len(non_code_switched))
print('Total number of utterances: ', len(code_switched)+len(non_code_switched))
# for utt in code_switched:
# 	words = [t[0] for t in utt]
# 	print(words)

train = code_switched[0:-2000] #+ non_code_switched
shuffle(train)

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

