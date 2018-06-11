import json
from random import shuffle

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data

	

def tag_data_w_source(data, source):
	switched = []
	non_switched = []
	for d in data:
		eng = False
		spa = False

		for token in d:
			lang = token[1]
			if lang == 'spa' or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa' or lang == 'mixed':
				spa = True
			if lang == 'eng'or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa'or lang == 'mixed':
				eng = True

		sourced = []
		for i in range(len(d)):
			word = d[i][0]
			lang = d[i][1]
			sourced.append((word, lang, source))

		if eng and spa: switched.append(sourced)
		else: non_switched.append(sourced)

	return switched, non_switched


miami_data = open_json('miami-data.json')
twitter_data = open_json('twitter-data.json')

miami_switched, miami_non_switched = tag_data_w_source(miami_data, 'miami')
twitter_switched, twitter_non_switched = tag_data_w_source(twitter_data, 'twitter')

print('Number of code-switched utterances: ', len(miami_switched) + len(twitter_switched))
print('Number of non-code-switched utterances: ', len(miami_non_switched) + len(twitter_non_switched))
print('Total number of utterances: ', len(miami_switched) + len(twitter_switched) + len(miami_non_switched) + len(twitter_non_switched))


data = miami_switched + twitter_switched
shuffle(data)

train = data[0:-2000]
val_test = data[-2000:]
val = val_test[:1000]
test = val_test[1000:]

print('Train size: ', len(train))
print('Val size: ', len(val))
print('Test size: ', len(test))

with open('../train.json', 'w') as outfile:
    json.dump(train, outfile)

with open('../val.json', 'w') as outfile:
    json.dump(val, outfile)

with open('../test.json', 'w') as outfile:
    json.dump(test, outfile)

