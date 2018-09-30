import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA

import seaborn as sns

data = pickle.load(open('Stack_api_data_processed.pkl', "rb" ))

def field_clustering():
	X = list(data.loc[:,'w2v'])
	labels = data.loc[:, 'field']
	pca = PCA(n_components=2)
	Ypca = pca.fit_transform(X)

	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = ['x', 'y']), labels], axis = 1)
	sns.lmplot(x='x', y='y', data=PCAdata, fit_reg=False, hue='field')
	plt.show()
#field_clustering()

def word_clustering():
	X = list(data.loc[:,'relevant keywords w2v'])
	v = []
	question_label = []
	for i in range(len(data)):
		x = X[i]
		for y in x:
			v.append(y)
			question_label.append(data.loc[i,'question_id'])
	labels = pd.DataFrame(question_label, columns = ['question'])
	pca = PCA(n_components=2)
	Ypca = pca.fit_transform(v)

	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = ['x', 'y']), labels], axis = 1)
	sns.lmplot(x='x', y='y', data=PCAdata, fit_reg=False, hue = 'question')
	plt.show()

def word_clustering():
	X = data.loc[:,'relevant keywords w2v']
	v = []
	question_label = []
	for i in range(len(X)):
		x = X[i]
		for y in x:
			v.append(y)
			question_label.append(data.loc[i,'question_id'])
	labels = pd.DataFrame(question_label, columns = ['question'])
	pca = PCA(n_components=2)
	Ypca = pca.fit_transform(v)

	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = ['x', 'y']), labels], axis = 1)
	sns.lmplot(x='x', y='y', data=PCAdata, fit_reg=False, hue = 'question', legend = False)
	plt.show()

word_clustering()


#print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  








