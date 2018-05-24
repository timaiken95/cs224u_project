
## ==================
## TO DO
## POS tags ???
## Encoding to ascii
## Start/End token added at each period?
## ==================

# Language markers 
# @s:spa = Spanish
# @s:eng = English
# @s:eng&spa = Undetermined
# @s:spa+eng = word with first morpheme(s) Spanish, second morpheme(s) English
# @s:eng+spa = word with first morpheme(s) English, second morpheme(s) Spanish
# Untagged words are English except where part of an utterance headed [- spa], in which untagged words are Spanish

import os
import json
from unidecode import unidecode

folder = 'miami-data'
files = os.listdir(folder)
word_remove = set(['', '\"', "'s", '"', '-', '=', 'xxx', 'x', '+<', '/-', '[/]', '[//]', '[///]', '+...', '+/.', '+//.', '+"/.', '+"', '+,', '+..?', '+".', '+//?', '+/?', '+/?', '++', '+.', '+!?'])
char_remove_table = dict.fromkeys(map(ord, '&!?()[]"<>.,/:*=-'), None)

data = []

vocabulary = set()

for file_name in files:
	if not file_name.endswith('.cha'): continue
	with open(folder + '/' + file_name) as f:

		while True:
			line = f.readline().strip()
			if not line: break

			## encode with ascii
			# line = unidecode(line)
			# line = line.encode('ascii', 'ignore')

			if line.startswith('@'): continue
			if line.startswith('%eng'): continue

			tags_line = f.readline().strip()
			if not tags_line.startswith('%aut'): continue ## then line is only a laugh/cough/etc

			tags = tags_line.split(':\t')[1]
			comment = line.split(':\t')[1]

			## determine default language
			if comment.find('[- spa]') > -1:
				default_lang = 'spa'
				index = comment.find('[- spa]') + 8
				comment = comment[index:]
			elif comment.find('[- eng]') > -1: 
				default_lang = 'eng'
				index = comment.find('[- eng]') + 8
				comment = comment[index:]
			else: 
				default_lang = 'eng'

			all_words = comment.split()
			pos = tags.split()

			words = []
			for w in all_words:
				w = w.strip()
				if w.find('\x15') > -1: continue
				if w.find('xxx') > -1: continue
				if w.find('&=') > -1: continue
				if w in word_remove: continue

				lang = default_lang
				if w.find('@s:') > -1:
					w_split = w.split('@s:')
					w = w_split[0]
					lang = w_split[1]

				w = w.translate(char_remove_table)
				w = w.lower()
				if w == '': continue

				if w.find('n\'t') > -1:
					w_split = w.split('n\'t')
					words.append([w_split[0], lang])
					words.append(['n\'t', lang])
					vocabulary.add(w_split[0])
					vocabulary.add('n\'t')
				elif w.find('\'') > -1:
					w_split = w.split('\'')
					words.append([w_split[0], lang])
					words.append(['\'' + w_split[1], lang])
					vocabulary.add(w_split[0])
					vocabulary.add('\'' + w_split[1])
				else: 
					words.append([w, lang])
					vocabulary.add(w)

			# if len(pos) != len(words): 
			# 	print(words)
			# 	print(pos)
			# 	print()

			# i = 0
			# while i < len(pos):
			# 	pos_split = pos[i].split('.', 1)
			# 	if len(pos_split) < 2: pos_tag = pos_split[0]
			# 	else: pos_tag = pos_split[1]
			# 	i += 1

			if len(words) >= 1:
				# start_lang = words[0][1]
				# end_lang = words[len(words)-1][1]
				# words = [['<start>', start_lang]] + words + [['<end>', end_lang]]
				data.append(words)
			
			
print(len(vocabulary))
print(len(data))
# print(sorted(list(vocabulary)))
# for v in vocabulary: print(v)

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)








