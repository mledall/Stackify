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

from Data_overflow_processing import semantic_distance_to_other


data = pickle.load(open('Stackoverflow_data_processed.pkl', "rb" ))
raw_data = pickle.load(open('Stackoverflow_data.pkl', "rb" ))

ex_link = 'https://stackoverflow.com/questions/30081275/why-is-1000000000000000-in-range1000000000000001-so-fast-in-python-3'


def user_input(link):
	index = np.where(data.loc[:,'link'] == link)[0][0]
	title = BeautifulSoup(raw_data.loc[index,'title'], 'html.parser').get_text()
	body = BeautifulSoup(raw_data.loc[index,'body'], 'html.parser').get_text()
	return index, title, body

def get_keywords(link):
	index = user_input(link)[0]
	keywords = data.loc[index, 'keywords']
	relevant_keywords = []
	for keyword, distance in data.loc[index, 'relevant keywords']:
		relevant_keywords.append(keyword)
	return keywords, relevant_keywords

def recommended_articles(keyword, link):
	index = user_input(link)[0]
	recommended = semantic_distance_to_other(keyword, data)
	recommended_redundant = []
	for idx, distance in recommended:
		if not idx == data.loc[index, 'id']:
			recommended_redundant.append((idx, distance))
	return recommended, recommended_redundant

def main_function(link):
	index, title, body = user_input(link)
	relevant_keywords = get_keywords(link)[1]
	reference_list = []
	for keyword in relevant_keywords:
		reference_id = recommended_articles(keyword, link)[1][0][0]
		reference_score = recommended_articles(keyword, link)[1][0][1]
		reference_link = data.loc[np.where(data.loc[:,'id'] == reference_id)[0][0], 'link']
		reference_list.append((keyword, reference_id, reference_link, reference_score))
	return reference_list

print(main_function(ex_link))

#relevant_keywords = get_keywords(ex_link)[1]
#top_keyword = relevant_keywords[0]
#print(recommended_articles(top_keyword, ex_link)[0][:10], recommended_articles(top_keyword, ex_link)[1][:10])
#print(np.where(data.loc[:,'id'] == 21530577))

#print(recommended_articles(ex_link))
#print(raw_data.loc[0,'link'])
#print(BeautifulSoup(raw_data.loc[0,'title']).get_text())







