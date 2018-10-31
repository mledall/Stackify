'''
This scripts is the pipeline for the Stackify app

1) Get the text that the user is reading
2) Extract the keywords from the text, along with their TextRank score and word average
3) For all of the keywords, fetch the 3 most relevant other articles based on the semantic distance between the word vectors and the question average vectors
4) Fetch the link for all of the vectors.

Because of the limitation of the app, currently I am unable to apply this pipeline to any StackExchange post. This is because I built a question database ahead of time and compare articles within that data base. 

To address this limitation, I need to scrape any StackExchange and apply the app to that new text. Doing so will allow me to be more general than just StackExchange.
'''

import pandas as pd
import pickle
import numpy as np
from bs4 import BeautifulSoup

data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))
raw_data = pickle.load(open('iterative_scraping_Stackoverflow.pkl', "rb" ))

def user_input(link):
	'''
	summarizes the information from the article being read by the user. The input link must belong to the database, this is a big limitation.
	'''
	index = list(raw_data[raw_data.loc[:, 'link'] == link].index)[0]
	question_id = raw_data.loc[index,'id']
	title = BeautifulSoup(raw_data.loc[index,'title'], 'html.parser').get_text()
	body = BeautifulSoup(raw_data.loc[index,'body'], 'html.parser').get_text()
	field = raw_data.loc[index,'field']
	return index, question_id, title, body, field

def semantic_distance(vec1, vec2):
	'''
	calculates a semantic distance based on the cosine similarity between vectors
	'''
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return cosine_similarity

def most_relevant_article(keyword_vec, data):
	'''
	creates a list of the most relevant questions given a specific word vector
	'''
	dummy_list = []
	for i in range(len(data)):
		w2v_average = data.loc[i, 'average w2v']
		if type(w2v_average) != int:
			w2v_distance = semantic_distance(keyword_vec, w2v_average)
			dummy_list.append([data.loc[i, 'id'], data.loc[i, 'link'], w2v_distance])
	return sorted(dummy_list, key=lambda distance: distance[-1], reverse = True)[:3]

def keyword_vectors(index):
	'''
	get the vectors for each keyword from the database
	'''
	keyword_dict = {}
	for word in data.loc[index, 'keywords w2v']:
		keyword = word[0]
		keyword_w2v = word[-1]
		keyword_dict[keyword] = keyword_w2v	
	return keyword_dict

def semantic_distance_to_other(question_id, field):
	index = list(data[data.loc[:, 'id'] == question_id].index)[0]
	other_articles = data[data.loc[:, 'id'] != question_id]
	other_articles_field = other_articles[other_articles.loc[:, 'field'] == field].reset_index()
	n_other_articles = len(other_articles_field)
	keyword_vecs = keyword_vectors(index)
	keyword_article_distance = {}
	for key in keyword_vecs:
		keyword_vec = keyword_vecs[key]
		relevant_article = most_relevant_article(keyword_vec, other_articles_field)
		keyword_article_distance[key] = relevant_article
	return keyword_article_distance

def main_function(link):
	'''
	Outputs the results in a formatted form
	'''
	index, question_id, title, body, field = user_input(link)
	relevant_keywords = semantic_distance_to_other(question_id, field)
	for key in relevant_keywords:
		print('{}:\n'.format(key))
		for keyword in relevant_keywords[key]:
			reference_link = keyword[1]
			reference_score = keyword[2]
			print(' - {}'.format(reference_link))
		print('\n')

#link = 'https://stackoverflow.com/questions/276761/exposing-a-c-api-to-python'
#main_function(link)



