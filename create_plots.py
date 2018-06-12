import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

both = np.load("full_data_no_language.npy").item()['f1'][0]
miami = np.load("miami_data_no_language.npy").item()['f1'][0]
twitter = np.load("twitter_data_no_language.npy").item()['f1'][0]

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Scores for Different Datasets")

red_patch = mpatches.Patch(color='red', label='Miami Dataset')
green_patch = mpatches.Patch(color='green', label='Twitter Dataset')
blue_patch = mpatches.Patch(color='blue', label='Combined Datasets')

x = np.arange(1, 41)
plt.plot(x, miami, 'r')
plt.plot(x, twitter, 'g')
plt.plot(x, both, 'b')

plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.savefig('datasets.png')

plt.close()

language = np.load("full_data_yes_language.npy").item()['f1'][0]
ner = np.load("full_data_yes_language_ner.npy").item()['f1'][0]
pos = np.load("full_data_yes_language_pos.npy").item()['f1'][0]
ner_pos = np.load("full_data_yes_language_pos_ner.npy").item()['f1'][0]

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Scores for Language,\n Language+POS, Language+NER, and All")

red_patch = mpatches.Patch(color='red', label='Language')
green_patch = mpatches.Patch(color='green', label='Language+POS')
blue_patch = mpatches.Patch(color='blue', label='Language+NER')
yellow_patch = mpatches.Patch(color='yellow', label='Language+POS+NER')

x = np.arange(1, 41)
plt.plot(x, language, 'r')
plt.plot(x, pos, 'g')
plt.plot(x, ner, 'b')
plt.plot(x, ner_pos, 'y')

plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch])

plt.savefig('tags.png')

plt.close()

lstm = np.load("full_data_yes_language.npy").item()['f1'][0]
rnn = np.load("rnn_yes_language.npy").item()['f1'][0]
gru = np.load("gru_yes_language.npy").item()['f1'][0]

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Scores for Different Recurrant Units")

red_patch = mpatches.Patch(color='red', label='LSTM')
green_patch = mpatches.Patch(color='green', label='Vanilla RNN')
blue_patch = mpatches.Patch(color='blue', label='GRU')

x = np.arange(1, 41)
plt.plot(x, lstm, 'r')
plt.plot(x, rnn, 'g')
plt.plot(x, gru, 'b')

plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.savefig('units.png')

plt.close()
