import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from Data_overflow_processing import semantic_distance_to_other

data = pickle.load(open('Stackoverflow_data_processed.pkl', "rb" ))

def semantic_distance_to_avg(index):
	distance_to_avg = []
	for word, distance in data.loc[index, 'relevant keywords']:
		distance_to_avg.append([word, distance])
	distance_to_avg_df = pd.DataFrame(distance_to_avg, columns = ['keyword','cosine similarity'])
	ax = sns.barplot(x='cosine similarity', y='keyword', data=distance_to_avg_df)
#	plt.xticks(rotation=45)
	plt.show()
#semantic_distance_to_avg(0)

def semantic_distance_to_other_articles(keyword):
	distance_to_other = list(semantic_distance_to_other(keyword)[:10])
	distance_to_other_df = pd.DataFrame(distance_to_other, columns = ['question id', 'cosine similarity'])
	ax = sns.barplot(x='question id', y='cosine similarity', data=distance_to_other_df)
#	plt.xticks(rotation=45)
	plt.show()
#top_keyword = data.loc[0, 'relevant keywords'][0][0]
#semantic_distance_to_other_articles(top_keyword)

def pca_type_clustering(n_components):	# This function clusters the articles
	X = list(data.loc[:,'w2v'])
	labels = data.loc[:, 'type']
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(X)

	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = ['PC1', 'PC2']), labels], axis = 1)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='type')
	plt.show()
#pca_type_clustering(2)

def pca_field_clustering(n_components):	# This function clusters the articles
	X = list(data.loc[:,'w2v'])
	types = data.loc[:, 'type']
	fields = data.loc[:, 'field']
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(X)
	cols = ['PC'+str(i) for i in range(1, n_components+1)]
	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = cols), fields, types], axis = 1)
#	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
#	fig, axs = plt.subplots(ncols=3)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field', legend = False)
	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field', legend = False)
	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field', legend = True)
#	fig = plt.figure()
#	ax = Axes3D(fig, elev=48, azim=134)
#	ax.scatter(Ypca[:, 0], Ypca[:, 1], Ypca[:, 2], c=labels, edgecolor='k')#.astype(np.float)
#	sns.pairplot(PCAdata, vars=['PC1','PC2', 'PC3'], hue = 'field', kind = 'reg')
	plt.show()
pca_field_clustering(3)


def pca_word_clustering(n_components):	# This function clusters the words using the PCA algorithm. This algorithm achieves relatively good success.
	X = data.loc[:,'relevant keywords w2v']
	v = []
	label = []
	for i in range(len(X)):
		x = X[i]
		for y in x:
			v.append(y)
			label.append(data.loc[i,'field'])
	labels = pd.DataFrame(label, columns = ['field'])
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(v)
	cols = ['PC'+str(i) for i in range(1, n_components+1)]
	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = cols), labels], axis = 1)
#	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
#	fig, axs = plt.subplots(ncols=3)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field')
	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field')
	plt.show()
#pca_word_clustering(3)


def pca_wordvsavg_clustering():
	word_w2v = []
	avg_w2v = []
	for i in range(len(data)):
		avg_w2v.append([data.loc[i, 'question_id'], data.loc[i, 'w2v'], data.loc[i, 'field'], 'avg'])
		relevant_words_w2v = data.loc[i, 'relevant keywords w2v']
		for w2v in relevant_words_w2v:
			word_w2v.append([data.loc[i, 'question_id'], w2v, data.loc[i, 'field'], 'word'])
	avg_w2v_df = pd.DataFrame(avg_w2v, columns = ['question id', 'w2v', 'field', 'type'])
	word_w2v_df = pd.DataFrame(word_w2v, columns = ['question id', 'w2v', 'field', 'type'])
	wordvsavg_df = pd.concat([avg_w2v_df, word_w2v_df], axis = 0)
	X = list(wordvsavg_df.loc[:,'w2v'])
	pca = PCA(n_components=2)
	Ypca = pca.fit_transform(X)
	Ypca_df = pd.DataFrame(Ypca, columns = ['PC1', 'PC2'])
	question_ids = pd.DataFrame(list(wordvsavg_df.loc[:, 'question id']), columns = ['question ids'])
	fields = pd.DataFrame(list(wordvsavg_df.loc[:, 'field']), columns = ['field'])
	types = pd.DataFrame(list(wordvsavg_df.loc[:, 'type']), columns = ['type'])
	PCAdata = pd.concat([Ypca_df, question_ids, fields, types], axis = 1)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='type', )
	plt.show()
#	print(PCAdata)


def kmeans_field_clustering(n_clusters):	# This function clusters the articles
	X = list(data.loc[:,'w2v'])
	labels = data.loc[:, 'field']
	kmeans = KMeans(n_clusters=n_clusters)
	Ykmeans = kmeans.fit_transform(X)
	cols = ['PC'+str(i) for i in range(1, n_clusters+1)]
#	kmeansdata = pd.DataFrame(Ykmeans, columns = cols)
#	kmeansdata = pd.concat([pd.DataFrame(Ykmeans, columns = cols), labels], axis = 1)
#	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
#	fig, axs = plt.subplots(ncols=3)
#	sns.lmplot(x='PC1', y='PC2', data=kmeansdata, fit_reg=False)
#	sns.lmplot(x='PC1', y='PC3', data=kmeansdata, fit_reg=False)
#	sns.lmplot(x='PC2',y='PC3', data=kmeansdata, fit_reg=False)
#	plt.show()
	fig = plt.figure()
	ax = Axes3D(fig, elev=48, azim=134)
	labels = kmeans.labels_
	ax.scatter(Ykmeans[:, 0], Ykmeans[:, 1], Ykmeans[:, 2], c=labels, edgecolor='k')
	plt.show()
	
#kmeans_field_clustering(8)




'''
def kmeans_wordvsavg_clustering():
	word_w2v = []
	avg_w2v = []
	for i in range(len(data)):
		avg_w2v.append([data.loc[i, 'question_id'], data.loc[i, 'w2v'], data.loc[i, 'field'], 'avg'])
		relevant_words_w2v = data.loc[i, 'relevant keywords w2v']
		for w2v in relevant_words_w2v:
			word_w2v.append([data.loc[i, 'question_id'], w2v, data.loc[i, 'field'], 'word'])
	avg_w2v_df = pd.DataFrame(avg_w2v, columns = ['question id', 'w2v', 'field', 'type'])
	word_w2v_df = pd.DataFrame(word_w2v, columns = ['question id', 'w2v', 'field', 'type'])
	wordvsavg_df = pd.concat([avg_w2v_df, word_w2v_df], axis = 0)
	question_ids = pd.DataFrame(list(wordvsavg_df.loc[:, 'question id']), columns = ['question ids'])
	fields = pd.DataFrame(list(wordvsavg_df.loc[:, 'field']), columns = ['field'])
	types = pd.DataFrame(list(wordvsavg_df.loc[:, 'type']), columns = ['type'])
	X = list(wordvsavg_df.loc[:,'w2v'])
	kmeans = KMeans(n_clusters=8)
	Ykmeans = kmeans.fit_transform(X)
	Ykmeans_df = pd.DataFrame(Ykmeans, columns = ['PC'+str(i) for i in range(8)])
	kmeansdata = pd.concat([Ykmeans_df, question_ids, fields, types], axis = 1)
	sns.lmplot(x='PC1', y='PC3', data=kmeansdata, fit_reg=False, hue='type')
	plt.show()
#	print(kmeansdata)

#kmeans_wordvsavg_clustering()



#print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  
'''

'''
def tsne_word_clustering(n_components, perplexity):	# This function clusters the words using the TSNE alogrithm. This algorithm does not achieve good clustering.
	X = data.loc[:,'relevant keywords w2v']
	v = []
	question_label = []
	for i in range(len(X)):
		x = X[i]
		for y in x:
			v.append(y)
			question_label.append(data.loc[i,'field'])
	labels = pd.DataFrame(question_label, columns = ['field'])
	tsne = TSNE(n_components=n_components, perplexity = perplexity)
	Ytsne = tsne.fit_transform(v)
#	print(pca.explained_variance_ratio_)
	TSNEdata = pd.concat([pd.DataFrame(Ytsne, columns = ['x', 'y']), labels], axis = 1)
	sns.lmplot(x='x', y='y', data=TSNEdata, fit_reg=True, hue = 'field', legend = True)
	plt.show()

#tsne_word_clustering(2, 10)

def pca_tsne_word_clustering():	# This function combines pca and tsne. We find no significant clustering improvement over using either tsne or pca alone.
	X = data.loc[:,'relevant keywords w2v']
	v = []
	question_label = []
	for i in range(len(X)):
		x = X[i]
		for y in x:
			v.append(y)
			question_label.append(data.loc[i,'field'])
	labels = pd.DataFrame(question_label, columns = ['field'])
	pca = PCA(n_components = 30)	# We first use PCA to reduce the dimensions to 50
	Ypca = pca.fit_transform(v)
	tsne = TSNE(n_components = 2, perplexity = 50)	# We then use tSNE to reduce the dimensions to 2
	Ytsne = tsne.fit_transform(Ypca)
	TSNEdata = pd.concat([pd.DataFrame(Ytsne, columns = ['x', 'y']), labels], axis = 1)
	sns.lmplot(x='x', y='y', data=TSNEdata, fit_reg=True, hue = 'field', legend = True)
	plt.show()
#pca_tsne_word_clustering()





def article_semantic_distance(index):	#This function plots the semantic distance between words in an article and the article average.
	distances_to_average = []
	article_id = []
	field = []
	relevant_keywords = data.loc[index, 'relevant keywords']
	for keyword, distance in relevant_keywords:
		distances_to_average.append(distance)
		article_id.append(data.loc[index, 'question_id'])
		field.append(data.loc[index, 'field'])
	distances_to_averagedf = pd.DataFrame(distances_to_average, columns = ['distances to avg'])
	article_id_df = pd.DataFrame(article_id, columns = ['question id'])
	field_df = pd.DataFrame(field, columns = ['field'])
	distances_df = pd.concat([distances_to_averagedf, article_id_df, field_df], axis = 1)
	return distances_df


def cat_plot_semantic_distances():	# It is hard to plot the categorical plot 
	distances_df_list = []
	for i in range(len(data)):
		distances_df_list.append(article_semantic_distance(i))
	distances_df = pd.concat(distances_df_list, axis = 0)
	sns.boxplot(x = 'field', y = 'distances to avg', data = distances_df[:4000])
	plt.show()
#	print(distances_df)

cat_plot_semantic_distances()
'''



