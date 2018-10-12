import requests
import json
import pandas as pd
import pickle
import time

def scraper(baseurl, key, query, item_type, field, cols, pickle_path):
	response = requests.get(baseurl+query+key)
	data_json = response.json()
	items = []
	for i in range(len(data_json['items'])):
		item = data_json['items'][i]
		items.append([item['question_id'], item['tags'], item['title'], item['body'], item['link'], item_type, field])
	items_df = pd.DataFrame(items, columns = cols)
	pickle.dump(items_df, open(pickle_path, "wb" ) )
	return items_df

def extend_data_base(baseurl, key, query, item_type, field, cols, pickle_path):
	raw_data = pickle.load(open(pickle_path, "rb" ))
	new_items_df = scraper(baseurl, key, query, item_type, field, cols, pickle_path)
	items_df = pd.concat([raw_data, new_items_df], axis = 0, ignore_index = True)
	pickle.dump(items_df, open(pickle_path, "wb" ) )
	return items_df

def question_or_answer(week_init, week_fin, item_type, field):
	if item_type == 'questions':
		if field == 'python':
			return '/2.2/questions?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!LR0Y.(TGRRDGqDYvGAwl1e'
		if field == 'biology':
			return '/2.2/questions?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&site=biology&filter=!LR0Y.(TGRRDGqDYvGAwl1e'
		if field == 'physics':
			return '/2.2/questions?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&site=physics&filter=!LR0Y.(TGRRDGqDYvGAwl1e'
	if item_type == 'answers':
		if field == 'python':
			return '/2.2/answers?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&site=stackoverflow&filter=!b1MMEcCtn7nRIq'
		if field == 'biology':
			return '/2.2/answers?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&site=biology&filter=!b1MMEcCtn7nRIq'
		if field == 'physics':
			return '/2.2/answers?pagesize=100&fromdate='+str(week_init)+'&todate='+str(week_fin)+'&order=desc&sort=votes&site=physics&filter=!b1MMEcCtn7nRIq'



def data_base_iterator(week_fin, pickle_path, item_type, field):
	day = 24*3600
	week = day*7
	key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
	baseurl = 'https://api.stackexchange.com'
	cols = ['id','tags','title','body', 'link', 'type', 'field']
#	week_fin = todate
	for i in range(5):
		week_init = week_fin - week
		query = question_or_answer(week_init, week_fin, item_type, field)
		print('Scraping from week {} to week {}'.format(week_init, week_fin))
		extend_data_base(baseurl, key, query, item_type, field, cols, pickle_path)
		week_fin = week_init
	summary = 'We scraped until week ' + str(week_fin)
	print(summary)
	pickle.dump(week_fin, open('scraping_summary.pkl', "wb" ) )

def main_scraping_function():
	day = 24*3600
	week = day*7
#	fromdate = 1372636800
#	todate = 1538352000-week
	pickle_path = 'iterative_scraping_Stackoverflow.pkl'
	todate = pickle.load(open('scraping_summary.pkl', "rb" ))
	print('scraping Physics questions and answers')
	data_base_iterator(todate, pickle_path, item_type = 'questions', field = 'physics')
	data_base_iterator(todate, pickle_path, item_type = 'answers', field = 'physics')
	print('scraping Biology questions and answers')
	data_base_iterator(todate, pickle_path, item_type = 'questions', field = 'biology')
	data_base_iterator(todate, pickle_path, item_type = 'answers', field = 'biology')
	print('scraping Python questions and answers')
	data_base_iterator(todate, pickle_path, item_type = 'questions', field = 'python')
	data_base_iterator(todate, pickle_path, item_type = 'answers', field = 'python')
#	raw_data = pickle.load(open(pickle_path, "rb" ))
#	print(raw_data.loc[:,'field'])

def initial_scraping():
	day = 24*3600
	week = day*7
	todate = 1538352000
	fromdate = todate - week
	key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
	baseurl = 'https://api.stackexchange.com'
	cols = ['id','tags','title','body', 'link', 'type', 'field']
	query = question_or_answer(fromdate, todate, 'questions', 'physics')
	pickle_path = 'iterative_scraping_Stackoverflow.pkl'
	scraper(baseurl, key, query, 'questions', 'physics', cols, pickle_path)

#initial_scraping()

#main_scraping_function()
pickle_path = 'iterative_scraping_Stackoverflow.pkl'
raw_data = pickle.load(open(pickle_path, "rb" ))
print(raw_data)



