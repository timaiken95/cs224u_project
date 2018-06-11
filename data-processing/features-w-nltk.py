import json

import nltk
# nltk.internals.config_java(options='-Xmx2G')
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger



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

### STANFORD NLP TAGGERS ###

# pos_path = 'stanford-nlp-models/stanford-postagger-full-2018-02-27/'
# ner_path = 'stanford-nlp-models/stanford-ner-2018-02-27/'
# spanish_core_ner_path = 'stanford-nlp-models/stanford-spanish-nlp/edu/stanford/nlp/models/ner/'

# english_pos = StanfordPOSTagger(pos_path + 'models/english-bidirectional-distsim.tagger', pos_path + 'stanford-postagger.jar', encoding='utf8')
# spanish_pos = StanfordPOSTagger(pos_path + 'models/spanish-distsim.tagger', pos_path + 'stanford-postagger.jar', encoding='utf8')

# english_ner = StanfordNERTagger(ner_path + 'classifiers/english.all.3class.distsim.crf.ser.gz', ner_path + 'stanford-ner.jar', encoding='utf8')
# spanish_ner = StanfordNERTagger(spanish_core_ner_path + 'spanish.ancora.distsim.s512.crf.ser.gz', ner_path + 'stanford-ner.jar', encoding='utf8')


to_process = code_switched
featurized_data = []

for d in to_process:
	sent = [w[0] for w in d]
	
	# e_pos_tags = pos_tag(sent)
	# e_pos_ner_tags = tree2conlltags(ne_chunk(e_pos_tags))

	# e_pos_tags = english_pos.tag(sent)
	# s_pos_tags = spanish_pos.tag(sent)
	# e_ner_tags = english_ner.tag(sent)
	# s_ner_tags = spanish_ner.tag(sent)
	
	featurized_d = []
	for i in range(len(d)):
		word = d[i][0]
		lang = d[i][1]

		# e_pos = e_pos_ner_tags[i][1]
		# e_ner = e_pos_ner_tags[i][2]

		# e_pos = e_pos_tags[i][1]
		# s_pos = s_pos_tags[i][1]
		# e_ner = e_ner_tags[i][1]
		# s_ner = s_ner_tags[i][1]
		# pos.add(e_pos)
		# ner.add(e_ner)
		# featurized_d.append((word, lang, e_pos, e_ner))

	featurized_data.append(featurized_d)

print('done processing...')
print('POS: ', pos)
print('NER: ', ner)

with open('featurized-data/code-switched-e-pos-ner.json', 'w') as outfile:
    json.dump(featurized_data, outfile)

