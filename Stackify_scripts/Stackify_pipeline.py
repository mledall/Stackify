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


data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))
raw_data = pickle.load(open('iterative_scraping_Stackoverflow.pkl', "rb" ))

ex_link = 'https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python'


def user_input(link):
	index = list(raw_data[raw_data.loc[:, 'link'] == link].index)[0]
	title = BeautifulSoup(raw_data.loc[index,'title'], 'html.parser').get_text()
	body = BeautifulSoup(raw_data.loc[index,'body'], 'html.parser').get_text()
	field = raw_data.loc[index,'field']
	question_id = raw_data.loc[index,'id']
	return title, body, field, question_id

def get_keywords(link):
	question_id = user_input(link)[-1]
	index = list(data[data.loc[:, 'id'] == question_id].index)[0]
	keywords = data.loc[index, 'keywords']
	relevant_keywords = []
	for keyword, distance in data.loc[index, 'relevant keywords']:
		relevant_keywords.append(keyword)
	return keywords, relevant_keywords, index


def main_function(link):
	title, body, field, question_id = user_input(link)
	relevant_keywords = get_keywords(link)[1]
	index = get_keywords(link)[-1]
	reference_list = []
	for keyword in relevant_keywords:
		recommend = semantic_distance_to_other(keyword, field, question_id, data)
		reference_id = recommend.loc[0,'id']
		reference_score = recommend.loc[0,'cosine similarity']
		reference_link = data.loc[np.where(data.loc[:,'id'] == reference_id)[0][0], 'link']
		print('Keyword {} to link {} with reference score {}\n'.format(keyword, reference_link, reference_score))
		reference_list.append((keyword, reference_id, reference_link, reference_score))
	return reference_list

#print(list(data[data.loc[:, 'link'] == ex_link].index)[0])
#print(data[data.loc[:, 'field'] == 'python']['link'][1])

main_function(ex_link)









