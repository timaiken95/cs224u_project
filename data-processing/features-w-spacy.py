import json
import spacy

print('creating taggers...')
english_nlp = spacy.load('en')
spanish_nlp = spacy.load('es')

pos_tag_list = set()
ner_tag_list = set()

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data

def get_switched(data):
	switched = []

	for d in data:
		eng = False
		spa = False

		for token in d:
			lang = token[1]
			if lang == 'spa' or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa' or lang == 'mixed':
				spa = True
			if lang == 'eng'or lang == 'eng&spa' or lang == 'spa+eng' or lang == 'eng+spa'or lang == 'mixed':
				eng = True

		if eng and spa: switched.append(d)

	return switched

def match_pos_ner(d, doc, source):
		pos_tup = [(w.text, w.pos_, 'None') for w in doc]
		ner_tup = [(e.text, e.start, e.end, e.label_) for e in doc.ents]

		# for _ in pos_tup: print(_)
		# print ''
		# for _ in ner_tup: print(_)
		# print ''
		# for _ in d: print(_)
		# print ''
		# print('\n\n')

		for _, p, _ in pos_tup: pos_tag_list.add(p)
		for _, _, _, n in ner_tup: ner_tag_list.add(n)

		## merge pos_tup and ner_tup
		if len(ner_tup) > 0:
			for _, start, end, label in ner_tup:
				for i in range(start, end):
					pos_tup[i] = (pos_tup[i][0], pos_tup[i][1], label)

		new_d = []
		d_i = 0
		p_i = 0
		lang = d[0][1]

		while p_i < len(pos_tup) and d_i < len(d):
			w, p, n = pos_tup[p_i]
			lang = d[d_i][1]

			if w == d[d_i][0]:
				d_i += 1
				p_i += 1
				new_d.append((w, lang, source, p, n))

			else:
				if (p_i + 1) < len(pos_tup):
					w_, p_, n_ = pos_tup[p_i + 1]
					new_d.append((w, lang, source, p, n))
					new_d.append((w_, lang, source, p_, n_))
				d_i += 1
				p_i += 2

		return new_d

		# if len(pos_tup) == len(d): 
		# 	for _ in d: print(_)
		# 	print ''
		# 	for _ in pos_tup: print(_)
		# 	print ''
		# 	for _ in new_d: print(_)
		# 	print '\n\n'

		# if not len(pos_tup) == len(d): 
		# 	for _ in d: print(_)
		# 	print ''
		# 	for _ in pos_tup: print(_)
		# 	print ''
		# 	for _ in new_d: print(_)
		# 	print '\n\n'

def merge_english_spanish(eng, spa):
	for e in eng: print(e)
	print ''
	for s in spa: print(s)
	print ''

	prev_w_eng = eng[0][0]
	prev_w_spa = spa[0][0]
	merged = []
	
	e_i = 0
	s_i = 0
	# for w_eng, lang, source, p_eng, n_eng in eng:
	while e_i < len(eng) and s_i < len(spa):

		w_eng, lang, source, p_eng, n_eng = eng[e_i]
		w_spa, _, _, p_spa, n_spa = spa[s_i]


		if (prev_w_eng == 'gon' and w_eng == 'na') or (prev_w_eng == 'got' and w_eng == 'ta') or (prev_w_eng == 'can' and w_eng == 'not'): 
			e_i += 1
			continue

		if (prev_w_spa == 'ima' and (w_eng == 'm' or w_eng == 'a')): 
			e_i += 1
			continue
		
		if w_eng == '\'' and w_spa == '\'': ## or p_spa == 'SYM' or p_eng == 'PUNCT':
			e_i += 1
			s_i += 1
			continue
		
		if not w_eng == '\'' and w_spa == '\'':
			s_i += 1
			w_spa, _, _, p_spa, n_spa = spa[s_i]

		if len(w_spa) > len(w_eng): w = w_spa
		else: w = w_eng
		
		merged.append((w, lang, source, p_eng, n_eng, p_spa, n_spa))
		
		s_i += 1
		e_i += 1
		prev_w_eng = w_eng
		prev_w_spa = w_spa

	for m in merged: print(m)
	print('\n\n')
	return merged

def get_tags(data, source):
	featurized_data = []

	for d in data:
		sent = ''
		for w, l in d: 
			sent += w + ' '
		
		doc_e = english_nlp(sent)
		doc_s = spanish_nlp(sent)

		english = match_pos_ner(d, doc_e, source)
		spanish = match_pos_ner(d, doc_s, source)

		merged = merge_english_spanish(english, spanish)
		featurized_data.append(merged)

	return featurized_data



miami_data = open_json('miami-data.json')
miami_switched = get_switched(miami_data)
miami_tagged = get_tags(miami_switched, 'miami')


twitter_data = open_json('twitter-data.json')
twitter_switched = get_switched(twitter_data)
twitter_tagged = get_tags(twitter_switched, 'twitter')

all_data_tagged = miami_tagged + twitter_tagged

with open('featurized-data/all-data-tagged.json', 'w') as outfile:
    json.dump(all_data_tagged, outfile)

with open('featurized-data/pos-tag-list.json', 'w') as outfile:
    json.dump(list(pos_tag_list), outfile)

with open('featurized-data/ner-tag-list.json', 'w') as outfile:
    json.dump(list(ner_tag_list), outfile)

print('done processing...')
