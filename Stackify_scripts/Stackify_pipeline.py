'''
This is going to be the pipeline of the stackify app

1) I get the text
2) I scrape the keywords
3) I order them in order of semantic relevance
4) I link to another article in relevance order
'''


import pandas as pd
import pickle
import numpy as np
from bs4 import BeautifulSoup

data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))
raw_data = pickle.load(open('iterative_scraping_Stackoverflow.pkl', "rb" ))

def user_input(link):
	index = list(raw_data[raw_data.loc[:, 'link'] == link].index)[0]
	question_id = raw_data.loc[index,'id']
	title = BeautifulSoup(raw_data.loc[index,'title'], 'html.parser').get_text()
	body = BeautifulSoup(raw_data.loc[index,'body'], 'html.parser').get_text()
	field = raw_data.loc[index,'field']
	return index, question_id, title, body, field

def semantic_distance(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return cosine_similarity

def most_relevant_article(keyword_vec, data):
	dummy_list = []
	for i in range(len(data)):
		w2v_average = data.loc[i, 'average w2v']
		if type(w2v_average) != int:
			w2v_distance = semantic_distance(keyword_vec, w2v_average)
			dummy_list.append([data.loc[i, 'id'], data.loc[i, 'link'], w2v_distance])
	return sorted(dummy_list, key=lambda distance: distance[-1], reverse = True)[:3]

def keyword_vectors(index):
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



