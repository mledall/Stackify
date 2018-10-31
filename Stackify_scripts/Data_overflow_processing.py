'''
This function takes the raw data and processes it. For each of the questions, this includes:
- Cleans the question text by removing stop words from the text, tokenizing, lemmatizing the words.

- Blends the question body and title together,

- Extracts the keywords from the text using TextRank

- Calculates the word vector for each keyword based on the word2vec vector embedding

- Calculates the average vector for each questions by average the word vectors of the keywords of the text.
 
'''

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
en_stop = set(nltk.corpus.stopwords.words('english'))

import spacy
from spacy.lang.en import English
import gensim
from gensim import corpora

from gensim.summarization import summarizer
from gensim.summarization import keywords

spacy.load('en')
parser = English()

def remove_html_stop(text):
	'''
	removes stop words and html text
	'''
	rm_html = BeautifulSoup(text, 'html.parser').get_text()	# removes html
	letters_only = re.sub("[^a-zA-Z]",           	# The pattern to search for; ^ means NOT
		                  " ",                   	# The pattern to replace it with
		                  rm_html )              	# The text to search
	lower_case = letters_only.lower()	         	# Convert to lower case
	words = lower_case.split()          	     	# Split into words
	stops = stopwords.words("english")
	stops.append('ve')
	stops = set(stops)
	#english_words = words.words()[1:100]
	meaningful_words = [w for w in words if not w in stops]	# Remove stop words from "words"
	return ' '.join(meaningful_words)			# Joins the words back together separated by

def tokenize(text):
	'''
	tokenizes the text
	'''
	lda_tokens = []
	tokens = parser(text)
	for token in tokens:
		if token.orth_.isspace():
		    continue
		elif token.like_url:
		    lda_tokens.append('URL')
		elif token.orth_.startswith('@'):
		    lda_tokens.append('SCREEN_NAME')
		else:
		    lda_tokens.append(token.lower_)
	return lda_tokens

def get_lemma(word):
	'''
	lemmatizes the words
	'''
	lemma = wn.morphy(word)
	if lemma is None:
		return word
	else:
		return lemma

def prepare_text(text):
	'''
	combines the tokenizing and lemmatizing steps
	'''
	tokens = tokenize(text)
	tokens = [token for token in tokens if len(token) > 2]
	tokens = [token for token in tokens if token not in en_stop]
	tokens = [get_lemma(token) for token in tokens]
	tokened_text = [token for token in tokens]
	tokened_text = ' '.join(tokened_text)
	return tokened_text

def clean_text(text):
	'''
	combines the stop word removal to the lemmatizing steps
	'''
	remove_stops = remove_html_stop(text)
	tokenized = prepare_text(remove_stops)
	return tokenized

def loading_w2v_model():
	'''
	Loads a pre-trained word2vec model on 1 billion news articles, trained by Google.
	'''
	print('load the w2v model')
	return pickle.load(open("word2vec_model.pkl", "rb" ))

def one_keyword_w2v(keyword):
	'''Calculates the word vector for one keyword. If the keyword is made of more than one word than belong to the w2v dictionary, the script splits the keywords in individual words and sums the word vectors. If the keyword does not belong to the w2v dictionary, the function ignores the keyword.'''
	keyword_split = keyword.split(' ')
	word_sum = []
	for i in range(len(keyword_split)):
		if keyword_split[i] in model.wv.vocab:
			word_sum.append(model[keyword_split[i]])
	word_sum = sum(word_sum)
	return word_sum

def many_keywords_w2v(text):	
	'''Iterates over the words in one question: extracts the keywords, their scores, and their word vectors.'''
	keyword_list = keywords(text, ratio=0.2, words=None, split=False, scores=True, pos_filter=None, lemmatize=True, deacc=True)
	keyword_list_w2v = []
	for keyword ,score in keyword_list:
		word_vector = one_keyword_w2v(keyword)
		if type(word_vector) != int:
			keyword_list_w2v.append([keyword, score, word_vector])
	return keyword_list_w2v

def averaging_w2v(list_keyword_w2v):
	'''
	Averages the keyword vectors into one question average vector.
	'''
	average = []
	N = len(list_keyword_w2v)
	for keyword_w2v in list_keyword_w2v:
		average.append(list(keyword_w2v)[-1])
	average = sum(average)
	if N > 0:
		average = average*N**(-1)
	return average

def cleaning_the_data():
	'''
	Processing the whole data base by iterating over each question
	'''
	mushed_vec = []
	keyword_w2v_vec = []
	average_w2v_vec = []
	for i in range(len(data)):
		data.loc[i,'title'] = clean_text(data.loc[i,'title'])
		data.loc[i,'body'] = clean_text(data.loc[i,'body'])
		mushed = data.loc[i,'title']+data.loc[i,'body']
		mushed_vec.append(mushed)
		list_keyword_w2v = many_keywords_w2v(mushed)
		keyword_w2v_vec.append(list(list_keyword_w2v))
		average_w2v = averaging_w2v(list(list_keyword_w2v))
		average_w2v_vec.append(average_w2v)
		if i % 200 == 0:
			print('processed entry {}'.format(i))
	data['mushed'] = mushed_vec
	data['keywords w2v'] = keyword_w2v_vec
	data['average w2v'] = average_w2v_vec

def data_processor():
	'''
	processes the data and pickles it
	'''
	print('cleaning the data')
	cleaning_the_data()
	print('randomizes the data')
	random_data = data.sample(frac=1, axis=0).reset_index(drop=True)
#	print(random_data['average w2v'])
	print('pickling the data')
	pickle.dump(random_data, open('iterative_scraping_Stackoverflow_processed.pkl', "wb"))

print('loading the raw data')
data = pickle.load(open('iterative_scraping_Stackoverflow.pkl', "rb" ))
#data = data[:1000]
model = loading_w2v_model()
data_processor()

#data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))
#print(data['mushed'])


