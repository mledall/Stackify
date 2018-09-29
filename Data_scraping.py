import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle

key = '&key=J3NfKzbKZxySqX5YDFuKJw(('
baseurl = 'https://api.stackexchange.com'
query_phys_question = '/2.2/questions?pagesize=100&order=desc&sort=votes&site=physics&filter=!3xJkL33p-JGrGDQCT'
query_phys_answer = '/2.2/answers?pagesize=100&order=desc&sort=votes&site=physics&filter=!azbR7x6_.0mcgH'
query_bio_question = '/2.2/questions?pagesize=100&order=desc&sort=votes&site=biology&filter=!3xJkL33p-JGrGDQCT'
query_bio_answer = '/2.2/answers?pagesize=100&order=desc&sort=votes&site=biology&filter=!azbR7x6_.0mcgH'

query_question = {'physics': query_phys_question, 'biology': query_bio_question}
query_answer = {'physics': query_phys_answer, 'biology': query_bio_answer}

response = {'physics': [requests.get(baseurl+query_question['physics']+key),requests.get(baseurl+query_answer['physics']+key)], 'biology': [requests.get(baseurl+query_question['biology']+key),requests.get(baseurl+query_answer['biology']+key)]}
#data = {'physics': response[].json()}
data_answer = {'physics': response['physics'][1].json(), 'biology': response['biology'][1].json()}
data_question = {'physics': response['physics'][0].json(), 'biology': response['biology'][0].json()}

questions = []
answers = []
fields = ['physics', 'biology']
#for field in fields:
#    for item in data_question[field]['items']:
#        questions.append([item['question_id'], item['tags'], item['title'], item['body'], field, 0])
#    for item in data_answer[field]['items']:
#        answers.append([item['question_id'], item['tags'], item['title'], item['body'], field, 1])
cols = ['question_id','tags','title','body', 'answer', 'field']
for i in range(len(fields)):
    field = fields[i]
    for j in range(len(data_question[field]['items'])):
        question = data_question[field]['items'][j]
        answer = data_answer[field]['items'][j]
        questions.append([question['question_id'], question['tags'], question['title'], question['body'], answer['body'], field])
df = pd.DataFrame(questions, columns = cols)
df.to_csv('Stack_api_data.csv')
pickle.dump(df, open('Stack_api_data.pkl', "wb" ) )



