# Language markers 
# @s:spa = Spanish
# @s:eng = English
# @s:eng&spa = Undetermined
# @s:spa+eng = word with first morpheme(s) Spanish, second morpheme(s) English
# @s:eng+spa = word with first morpheme(s) English, second morpheme(s) Spanish
# Untagged words are English except where part of an utterance headed [- spa], in which untagged words are Spanish

import os

folder = 'miami-data'
files = os.listdir(folder)

for file_name in files:
	if not file_name.endswith('.cha'): continue
	with open(folder + '/' + file_name) as f:



		while True:
			line = f.readline().strip()
			if not line: break

			if line.startswith('@'): continue
			if line.startswith('%eng'): continue

			pos_line = f.readline().strip()
			if not pos_line.startswith('%aut'): continue ## then line is only a laugh/cough/etc

			pos = pos_line.split(':\t')[1]
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

			words = comment.split()
			print(words)











