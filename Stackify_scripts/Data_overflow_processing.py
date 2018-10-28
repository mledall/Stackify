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

def loading_w2v_model():
	print('load the w2v model')
	return pickle.load(open("word2vec_model.pkl", "rb" ))

def one_keyword_w2v(keyword):
	''' Function that calculates the word vector for a given keyword. If the keyword is made of two words that do not belong to the w2v dictionary, the script splits the keywords in individual words and sums the word vectors.'''
	keyword_split = keyword.split(' ')
	word_sum = []
	for i in range(len(keyword_split)):
		if keyword_split[i] in model.wv.vocab:
			word_sum.append(model[keyword_split[i]])
	word_sum = sum(word_sum)
	return word_sum

def many_keywords_w2v(text):	
	'''This function takes a paragraph as input and calculates all of the keywords, their scores, and their word vectors.'''
	keyword_list = keywords(text, ratio=0.2, words=None, split=False, scores=True, pos_filter=None, lemmatize=True, deacc=True)
	keyword_list_w2v = []
	for keyword ,score in keyword_list:
		word_vector = one_keyword_w2v(keyword)
		if type(word_vector) != int:
			keyword_list_w2v.append([keyword, score, word_vector])
	return keyword_list_w2v

def averaging_w2v(list_keyword_w2v):
	average = []
	N = len(list_keyword_w2v)
	for keyword_w2v in list_keyword_w2v:
		average.append(list(keyword_w2v)[-1])
	average = sum(average)
	if N > 0:
		average = average*N**(-1)
	return average

def cleaning_the_data():
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

def semantic_distance(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return cosine_similarity

def semantic_distance_to_other(keyword, field, question_id, data):
	semantic_distances = []
	for i in range(len(data)):
		semantic_distances.append([data.loc[i, 'id'], semantic_distance(data.loc[i,'average w2v'], model[keyword]), data.loc[i, 'field'], data.loc[i, 'keywords w2v']])
	distance_to_other = pd.DataFrame(semantic_distances, columns = ['id', 'cosine similarity', 'field', 'relevant keywords'])
	other_articles = distance_to_other[distance_to_other.loc[:, 'id'] != question_id]
	other_articles_fields = other_articles[other_articles.loc[:, 'field'] == field]
	other_articles_fields = other_articles_fields.sort_values(by = ['cosine similarity'], ascending = False)
	return other_articles_fields.reset_index()



'''
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

def cleaning_the_data():
	for i in range(len(data)):
		data.loc[i,'title'] = clean_text(data.loc[i,'title'])
		data.loc[i,'body'] = clean_text(data.loc[i,'body'])
		print('cleaned entry {}'.format(i))
'''

'''
def averaging_w2v():
	averages = []
	for question_w2v in data.loc[:,'keywords w2v']:
		average = []
		N = len(question_w2v)
		for keyword_w2v in question_w2v:
			average.append(keyword_w2v[1])
		average = sum(average)
		average = average*N**(-1)
		averages.append(average)
	return averages
#	data['average w2v'] = averages

'''

'''
def doc_w2v_average(text):
    tokens = tokenize(clean_text(text))
    average = np.array([0 for i in range(300)])
    it = 0
    for token in tokens:
        if token in model.wv.vocab:
            average = average + model[token]
            it+=1
    return average
'''

'''
data, cols = loading_raw_data()
data = data[:10]
def cleaning_data_test():
	for i in range(len(data)):
		data.loc[i, 'body'] = clean_text(data.loc[i, 'body'])
		data.loc[i, 'title'] = clean_text(data.loc[i, 'title'])
		print('finished point {}'.format(i))

cleaning_data_test()
print(data.loc[0, 'body'])
#data_processor(data[:5])
#print(data['title'][:5])
'''

'''
def averaging_w2v(data):
	averages = []
	for text in data.loc[:,'mushed']:
		averages.append(doc_w2v_average(text)) 
	data['w2v'] = averages

'''


