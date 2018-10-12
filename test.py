import requests
import json
import pandas as pd
import pickle
import time

'''
		python_questions = '/2.2/questions?pagesize=100&fromdate='+str(1372636800)+'&todate='+str(1538352000)+'&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!LR0Y.(TGRRDGqDYvGAwl1e'
'''

'''
key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
baseurl = 'https://api.stackexchange.com'
query_questions = '/2.2/questions?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!LR0Y.(TGRRDGqDYvGAwl1e'

query_answers = '/2.2/answers?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=stackoverflow&filter=!b1MMEcCtn7nRIq'
'''

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

def extend_data_base(baseurl, key, python_questions, item_type, field, cols, pickle_path):
	raw_data = pickle.load(open(pickle_path, "rb" ))
	new_items_df = scraper(baseurl, key, python_questions, item_type, field, cols, pickle_path)
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
		questions = question_or_answer(week_init, week_fin, item_type, field)
		print('Scraping from week {} to week {}'.format(week_init, week_fin))
		extend_data_base(baseurl, key, questions, item_type, field, cols, pickle_path)
		week_fin = week_init
	summary = 'We scraped until week ' + str(week_fin)
	print(summary)
	pickle.dump(week_fin, open('scraping_summary.pkl', "wb" ) )

fromdate = 1372636800
todate = 1538352000

week_fin = pickle.load(open('scraping_summary.pkl', "rb" ))
data_base_iterator(week_fin, 'test_automatic_scraping.pkl', item_type = 'questions', field = 'biology')
raw_data = pickle.load(open('test_automatic_scraping.pkl', "rb" ))
print(raw_data.loc[:,'field'])








