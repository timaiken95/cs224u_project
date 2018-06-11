import json
import spacy


## Useful documentation
## http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
## Twitter English tagger: https://gate.ac.uk/wiki/twitter-postagger.html

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data

miami_data = open_json('miami-data.json')
twitter_data = open_json('twitter-data.json')
data = miami_data + twitter_data

ner = set()
pos = set()
code_switched = []

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

print('creating taggers...')

to_process = code_switched
featurized_data = []

for d in to_process:
	sent = [w[0] for w in d]

	
	featurized_d = []
	for i in range(len(d)):
		word = d[i][0]
		lang = d[i][1]

		
		pos.add(e_pos)
		ner.add(e_ner)
		# featurized_d.append((word, lang, e_pos, e_ner))

	featurized_data.append(featurized_d)

print('done processing...')
print('POS: ', pos)
print('NER: ', ner)

with open('featurized-data/code-switched-e-pos-ner.json', 'w') as outfile:
    json.dump(featurized_data, outfile)

