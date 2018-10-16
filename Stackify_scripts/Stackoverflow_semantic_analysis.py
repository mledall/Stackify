import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from Data_overflow_processing import semantic_distance_to_other
from text_graph import get_graph

from gensim.summarization import keywords, graph
import networkx as nx

data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))

def graphing_text(text):
	G = get_graph(text)
	nodes = G.nodes()
	edges = G.edges()
	g = nx.Graph()	# Using networkx to define the graph will help us use the handy network plotting utilities
	for node in nodes:
		g.add_node(node)
	for edge in edges:
		g.add_edge(edge[0], edge[1])
	pos=nx.spring_layout(g)
	nx.draw_networkx_edges(g,pos, width=2)
	nx.draw_networkx_nodes(g,pos,node_size=100)
	nx.draw_networkx_labels(g,pos,font_size=15,font_family='sans-serif')
	plt.axis('off')
	plt.show()

text = 'Challenges in natural language processing frequently involve speech recognition, natural language understanding, natural language generation (frequently from formal, machine-readable logical forms), connecting language and machine perception, dialog systems, or some combination thereof.'

graphing_text(text)

def semantic_distance_to_avg(index):
	distance_to_avg = []
	for word, distance in data.loc[index, 'relevant keywords']:
		distance_to_avg.append([word, distance])
	distance_to_avg_df = pd.DataFrame(distance_to_avg, columns = ['keyword','cosine similarity'])
	ax = sns.barplot(x='cosine similarity', y='keyword', data=distance_to_avg_df)
	plt.show()
#semantic_distance_to_avg(0)

def semantic_distance_to_other_articles(keyword, question_id, field):
	distance_to_other = semantic_distance_to_other(keyword, field, question_id, data)
	vector = []
	it = 0
	for cos in distance_to_other.loc[:, 'cosine similarity']:
		it = it+1
		vector.append([it, cos])
	vector_df = pd.DataFrame(vector, columns = ['iterator', 'cosine similarity'])
	ax = sns.barplot(x='iterator', y='cosine similarity', data = vector_df[:10])
	plt.show()

#top_keyword = data.loc[0, 'relevant keywords'][0][0]
#field = data.loc[0, 'field']
#idx = data.loc[0, 'id']
#semantic_distance_to_other_articles(top_keyword, idx, field)


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
	fields = data.loc[:, 'field']
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(X)
	cols = ['PC'+str(i) for i in range(1, n_components+1)]
	PCAdata = pd.concat([pd.DataFrame(Ypca, columns = cols), fields], axis = 1)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field', legend = False)
	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field', legend = False)
	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field', legend = True)
	plt.show()
#print('performing pca on field data')
#pca_field_clustering(3)


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
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field')
	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field')
	plt.show()
#print('performing pca on word data')
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


def kmeans_with_tfidf(num_clusters = 3):
	'''
	I am following this one here http://brandonrose.org/clustering
	'''

	documents = list(data.loc[:,'body'])

	tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000, min_df=0.01, stop_words='english', use_idf=True, ngram_range=(1,3)) # 
	tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
	terms = tfidf_vectorizer.get_feature_names()

	km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
	model = km.fit(tfidf_matrix)
	Ykm = km.fit_transform(tfidf_matrix)
	clusters = pd.DataFrame(km.labels_.tolist(), columns = ['cluster'])

	cols = ['PC'+str(i) for i in range(1, num_clusters+1)]
	kmeansdata = pd.DataFrame(Ykm, columns = cols)
	kmeansdata = pd.concat([kmeansdata, clusters], axis = 1)

	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = tfidf_vectorizer.get_feature_names()
	for i in range(num_clusters):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
		    print(' %s' % terms[ind]),
		print
	print(kmeansdata['cluster'].value_counts())
	sns.lmplot(x='PC1', y='PC2', data=kmeansdata, fit_reg=False, hue = 'cluster', legend = False)
	sns.lmplot(x='PC1', y='PC3', data=kmeansdata, fit_reg=False, hue = 'cluster', legend = False)
	sns.lmplot(x='PC2',y='PC3', data=kmeansdata, fit_reg=False, hue = 'cluster', legend = True)
	plt.show()

#kmeans_with_tfidf()


