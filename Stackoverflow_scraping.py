# Using the Stackoverflow API, the https://api.stackexchange.com/docs/questions#pagesize=100&fromdate=2013-07-01&todate=2018-10-01&order=desc&sort=votes&tagged=python&filter=!LR0Y.(TGRRDGqDYvGAwl1e&site=stackoverflow&run=true

import requests
import json
import pandas as pd
import pickle

'''
key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
baseurl = 'https://api.stackexchange.com'
query_questions = '/2.2/questions?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!LR0Y.(TGRRDGqDYvGAwl1e'

query_answers = '/2.2/answers?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=stackoverflow&filter=!b1MMEcCtn7nRIq'
'''

def scraper(baseurl, key, query, item_type, field, cols):
	response = requests.get(baseurl+query+key)
	data_json = response.json()
	items = []
	for i in range(len(data_json['items'])):
		item = data_json['items'][i]
		items.append([item['question_id'], item['tags'], item['title'], item['body'], item['link'], item_type, field])
	items_df = pd.DataFrame(items, columns = cols)
	return items_df

key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
baseurl = 'https://api.stackexchange.com'
python_questions = '/2.2/questions?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!LR0Y.(TGRRDGqDYvGAwl1e'

python_answers = '/2.2/answers?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=stackoverflow&filter=!b1MMEcCtn7nRIq'

biology_questions = '/2.2/questions?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=biology&filter=!LR0Y.(TGRRDGqDYvGAwl1e'

biology_answers = '/2.2/answers?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=biology&filter=!b1MMEcCtn7nRIq'

physics_questions = '/2.2/questions?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=physics&filter=!LR0Y.(TGRRDGqDYvGAwl1e'

physics_answers = '/2.2/answers?pagesize=100&fromdate=1372636800&todate=1538352000&order=desc&sort=votes&site=physics&filter=!b1MMEcCtn7nRIq'

cols = ['id','tags','title','body', 'link', 'type', 'field']

#questions_df = scraper(baseurl, key, query_answers, 'answers', cols)
#print(questions_df.keys())


python_questions_df = scraper(baseurl, key, python_questions, 'questions', 'python', cols)
python_answers_df = scraper(baseurl, key, python_answers, 'answers', 'python', cols)
biology_questions_df = scraper(baseurl, key, biology_questions, 'questions', 'biology', cols)
biology_answers_df = scraper(baseurl, key, biology_answers, 'answers', 'biology', cols)
physics_questions_df = scraper(baseurl, key, physics_questions, 'questions', 'physics', cols)
physics_answers_df = scraper(baseurl, key, physics_answers, 'answers', 'physics', cols)



posts_df = pd.concat([python_questions_df, python_answers_df, physics_questions_df, physics_answers_df, biology_questions_df, biology_answers_df], axis = 0, ignore_index = True)


#posts_df.to_csv('Stackoverflow_data.csv')
pickle.dump(posts_df, open('Stackoverflow_data.pkl', "wb" ) )




