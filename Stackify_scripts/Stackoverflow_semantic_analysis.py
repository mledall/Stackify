'''
This scripts collects a number of plotting functions in order to understand our data and the workings of our algorithm

- graph out of the text
- ranking of the keywords based on TextRank score
- ranking of the articles based on the word2vec semantic similarity score
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from Stackify_pipeline import user_input, keyword_vectors, most_relevant_article
from text_graph import get_graph

from gensim.summarization import keywords, graph
import networkx as nx


data = pickle.load(open('iterative_scraping_Stackoverflow_processed.pkl', "rb" ))
link = 'https://stackoverflow.com/questions/276761/exposing-a-c-api-to-python'

question_id, title, body, field = user_input(link)[1:]
index = list(data[data.loc[:, 'link'] == link].index)[0]

def graphing_text():
	text = data.loc[index, 'mushed']
	G = get_graph(text)
	nodes = G.nodes()
	edges = G.edges()
	g = nx.Graph()	# Using networkx to define the graph will help us use the handy network plotting utilities
	for node in nodes:
		g.add_node(node)
	for edge in edges:
		g.add_edge(edge[0], edge[1])
	pos=nx.spring_layout(g)
	pos_adjust = {}
	for key in pos:
#		radius = np.linalg.norm(pos[key])
		pos_adjust[key] = pos[key]*1.1
	nx.draw_networkx_edges(g,pos, width=0.5)
	nx.draw_networkx_nodes(g,pos,node_size=50)
	nx.draw_networkx_labels(g,pos_adjust,font_size=15,font_family='sans-serif')
	plt.axis('off')
#	plt.savefig('text_graph.png')
	plt.tight_layout()
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.show()

#graphing_text()

def keyword_ranking():
	keyword_list = data.loc[index, 'keywords w2v']
	keyword_df = pd.DataFrame(keyword_list, columns = ['keyword', 'score', 'word vec'])
	sns.barplot(x = 'score', y = 'keyword', data = keyword_df)
	plt.title('TextRank score: ranking of words')
#	plt.savefig('top_keywords.png')
	plt.show()

#keyword_ranking()

def article_ranking(follow_up = 'api'):
	other_articles = data[data.loc[:, 'id'] != question_id]
	other_articles_field = other_articles[other_articles.loc[:, 'field'] == field].reset_index()
	keyword_vec = keyword_vectors(index)[follow_up]
	relevant_articles = most_relevant_article(keyword_vec, other_articles_field)
	it = 0
	vec = []
	for art in relevant_articles:
		vec.append([it, art[-1]])
		it+=1
	vec_df = pd.DataFrame(vec, columns = ['Iterator', 'cosine similarity'])
	ax = sns.barplot(x='Iterator', y='cosine similarity', data = vec_df)
	plt.show()

#article_ranking()

def pca_words(n_components = 2):
	Y = []
	labels = []
	it = 0
	for i in range(len(data)):
		entry = data.loc[i, 'keywords w2v']
		it += 1
		for word in entry:
			Y.append(word[-1])
			labels.append(data.loc[i, 'field'])
		if it%1000==0:
			print('finished entry {}'.format(it))
	print('pca transforming data')
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(Y)
	cols = ['PC'+str(i) for i in range(1, n_components+1)]
	print('making a dataframe out of the pca\'d data')
	Y_df = pd.DataFrame(Ypca, columns = cols)
	labels_df = pd.DataFrame(labels, columns = ['field'])
	PCAdata = pd.concat([Y_df, labels_df], axis = 1)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field')
#	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
#	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field')
	plt.show()
#pca_words()

def pca_fields(n_components = 2):	# This function clusters the articles
	entries = data['average w2v'].values
	Y = []
	labels = []
	it = 0
	for entry in entries:
		if type(entry) != int:
			Y.append(entry)
			labels.append(data.loc[it, 'field'])
		it+=1
	print('pca transforming data')
	pca = PCA(n_components=n_components)
	Ypca = pca.fit_transform(Y)
	cols = ['PC'+str(i) for i in range(1, n_components+1)]
	print('making a dataframe out of the pca\'d data')
	Y_df = pd.DataFrame(Ypca, columns = cols)
	labels_df = pd.DataFrame(labels, columns = ['field'])
	PCAdata = pd.concat([Y_df, labels_df], axis = 1)
	sns.lmplot(x='PC1', y='PC2', data=PCAdata, fit_reg=False, hue='field')
#	sns.lmplot(x='PC1', y='PC3', data=PCAdata, fit_reg=False, hue='field')
#	sns.lmplot(x='PC2',y='PC3', data=PCAdata, fit_reg=False, hue='field')
	plt.show()
pca_fields()



