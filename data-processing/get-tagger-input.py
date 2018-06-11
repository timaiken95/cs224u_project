import json

def open_json(file_name):
	data = []
	with open(file_name) as f:
	    data = json.load(f)
	return data

miami_data = open_json('miami-data.json')
twitter_data = open_json('twitter-data.json')
data = miami_data + twitter_data


code_switched = []
# non_code_switched = []

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
	# else: non_code_switched.append(d)

with open('code-switched-no-features.json', 'w') as outfile:
    json.dump(code_switched, outfile)

to_tag = open('to_tag.txt', 'w')

for item in code_switched:
	s = ''
	for w, l in item:
		s += w + ' '
	to_tag.write("%s\n" % s)