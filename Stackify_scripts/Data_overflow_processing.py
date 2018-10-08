import pandas as pd
#pd.options.display.float_format = '{:.4f}'.format

import numpy as np
from bs4 import BeautifulSoup
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet as wn
#nltk.download('wordnet')
#nltk.download('stopwords')
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

def remove_html_stop(text):		# Removes the HTML, punctuation, numbers, stopwords...
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

def tokenize(text): # the Tokenize function takes a text and splits it into its words
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
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokened_text = [token for token in tokens]
    tokened_text = ' '.join(tokened_text)
    return tokened_text

def clean_text(text):
    remove_stops = remove_html_stop(text)
    tokenized = prepare_text(remove_stops)
    return tokenized

def cleaning_the_data(data):
	for i in range(len(data)):
		data.loc[i,'title'] = clean_text(data.loc[i,'title'])
		data.loc[i,'body'] = clean_text(data.loc[i,'body'])


def mushing_text_data(data):
	mushed = data.loc[:,'title']+data.loc[:,'body']
	mush = []
	for text in mushed:
		mush.append(text)
	data['mushed'] = mush

#path = './GoogleNews-vectors-negative300.bin.gz'
#model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, limit = 1000000)
#pickle.dump(model, open('word2vec_model.pkl', "wb" ) )	

def loading_w2v_model():
	print('load the w2v model')
	model = pickle.load(open("word2vec_model.pkl", "rb" ))
#loading_w2v_model()

def doc_w2v_average(text):
    tokens = tokenize(clean_text(text))
    average = np.array([0 for i in range(300)])
    it = 0
    for token in tokens:
        if token in model.wv.vocab:
            average = average + model[token]
            it+=1
    return average

def averaging_w2v(data):
	averages = []
	for text in data.loc[:,'mushed']:
		averages.append(doc_w2v_average(text)) 
	data['w2v'] = averages

def grammatical_keywords(data):
	gram_keywords = []
	for mush in data.loc[:,'mushed']:
		gram_keywords.append(keywords(mush).split('\n'))
	data['keywords']=gram_keywords

def semantic_distance(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return cosine_similarity

def relevant_keywords(gram_keywords, average):
    relevant_keywords = []
    for word in gram_keywords:
        if word in model.wv.vocab:
            relevant_keywords.append((word, semantic_distance(average, model[word])))
    relevant_keywords.sort(key=lambda x: x[1], reverse = True)
    return relevant_keywords

def ranking_keywords(data):
	rank_keywords = []
	for i in range(len(data.loc[:,'keywords'])):
		gram_keywords = data.loc[i,'keywords']
		average = data.loc[i,'w2v']
		rank_keywords.append(relevant_keywords(gram_keywords,average))
	data['relevant keywords'] = rank_keywords
	return rank_keywords

def relevant_keywords_w2v(data):
	ranked_keywords = ranking_keywords(data)
	v = []
	for i in range(len(data)):
		relevant_keywords_w2v = []
		for keywords in ranked_keywords[i]:
			relevant_keywords_w2v.append(model[keywords[0]])
		v.append(relevant_keywords_w2v)
	data['relevant keywords w2v'] = v
	return v

def data_randomizer(data):
	return data.sample(frac=1, axis=0).reset_index(drop=True)

def loading_raw_data():
	print('loading the raw data')
	raw_data = pickle.load(open('iterative_scraping_Stackoverflow.pkl', "rb" )) #'Stackoverflow_data.pkl'
	raw_cols = list(raw_data.keys())
	return raw_data, raw_cols
#raw_data, raw_cols = loading_raw_data():

def data_processor(data):
	print('cleaning the data')
	cleaning_the_data(data)
	print('mushing the data')
	mushing_text_data(data)
	print('taking the word embedding average for each post')
	averaging_w2v(data)
	print('extracting the keywords for each post')
	grammatical_keywords(data)
	print('ranking the keywords for each post')
	ranking_keywords(data)
	print('taking the relevant keywords')
	relevant_keywords_w2v(data)
	print('randomizes the data')
	random_data = data_randomizer(data)
	print('pickling the data')
	pickle.dump(random_data, open('iterative_scraping_Stackoverflow_processed.pkl', "wb" ) ) #'Stackoverflow_data_processed.pkl'
#data_processor(raw_data)

def semantic_distance_to_other(keyword, field, question_id, data):
	semantic_distances = []
	for i in range(len(data)):
		semantic_distances.append([data.loc[i, 'id'], semantic_distance(data.loc[i,'w2v'], model[keyword]), data.loc[i, 'field'], data.loc[i, 'relevant keywords']])
	distance_to_other = pd.DataFrame(semantic_distances, columns = ['id', 'cosine similarity', 'field', 'relevant keywords'])
	other_articles = distance_to_other[distance_to_other.loc[:, 'id'] != question_id]
	other_articles_fields = other_articles[other_articles.loc[:, 'field'] == field]
	other_articles_fields = other_articles_fields.sort_values(by = ['cosine similarity'], ascending = False)
	return other_articles_fields.reset_index()

#data.to_csv('Stack_api_data_processed.csv')


