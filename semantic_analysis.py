import pandas as pd
import pickle

from Data_processing import semantic_distance_to_other

data = pickle.load(open('Stack_api_data_processed.pkl', "rb" ))
#model = pickle.load(open("word2vec_model.pkl", "rb" ))

print(semantic_distance_to_other(data.loc[1, 'relevant keywords'][0][0]))
#print(data.loc[:, 'relevant keywords'][0][0])

#semantic_distance_to_other(keyword)


