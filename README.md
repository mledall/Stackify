This is Stackify, a recommendation product that suggests semantically related articles by identifying key concepts in a Stackoverflow posting.

StackExchange is a Question&Answer based support platform, where users can ask technical questions, and other users can provide answers. StackExchange owns several subsidiary topic-specific support sites, including StackOverflow (specfic to computer science), Physics StackExchange (specific to physics), Biology StackExchange (specific to biology), which are few examples among tens of other sites, https://stackexchange.com/sites.

Readers typically go on the site to seek answers to specific questions, and learning from other people questions can be one of the most powerful ways to learn. Stackify aims to recommend related questions allowing the user to explore various facets of one topic. Utlitmately, users will spend more time on the platform thus increasing customer activity which is good for the business.

The subrepository, Stackify_scripts, contains all of the scripts necessary to achieve that, and here is a breakdown of the scripts,

- Iterative_scraping.py: Here, we use the StackExchange web API (Application Programming Interface) to scrape questions. The API allows up to 10,000 requests a day with 100 posts per request. For each of our requests, we ask the 100 most popular questions in a window of 3 weeks, and iterate on that for the past few years. Eventually, we collected over 40,000 questions. At this stage, the raw data collected contains the question title, the question body, the tags, the topic (physics, biology, python).

- Data_ overflow_ processing.py: Here, we process the raw data that was collected in Iterative_scraping.py. Since the raw data mostly contains text, we process the text by removing stop words, lemmatizing and stemming the text. Further, we perform feature engineering to create more input we will learn from, namely, we use TextRank to identify the keywords in each question, and further use Word2Vec word embedding to create word vectors for each of the question.

- Stackoverflow_ semantic_ analysis.py: Here, we use the features extracted in the previous script to explore our data. In particular, we use PCA (Principal Component Analysis) to visualize the distribution of words and questions. In addition, given one question as input, we are able to identify the relevant keywords, and for each keyword, we are able to recommend releted questions based on the semantic distance provided by Word2Vec.

- Stackify_pipeline.py: Here, put all of the pieces together from the user input to the product output. First, we get the topic, the question title, the question body and the question ID. Next, we identify keywords and provide the links to the related articles for each keyword. Finally, we provide this to the user presented as a list of keywords along with the links.






