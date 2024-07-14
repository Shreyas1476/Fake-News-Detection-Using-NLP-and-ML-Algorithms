# Fake-News-Detection-Using-NLP-and-ML-Algorithms
In this project we have used the various Machine Learning classification algorthims along with Natural Language Processing (NLP) for Fake News Detection.
We have used NLP in order to better utilize and understand text meaning and feature identification for fraud news articles. Machine learning-based misinformation detection systems might prove to be a very important tool for combating this problem.

All datasets are downloaded from Kaggle.The datasets used in this project are mentioned below :-

# Datasets used -

+ ### Fake News Detection Dataset :-

This dataset contains 12 columns and 22635 rows.Column titles include column titles: author, title, site_url, language, title_without_stopwords, text, published, main_img, type, label, text_without_stopwords, and hasImage. The data consists of news articles from various authors and types.The link for the dataset is https://www.kaggle.com/code/sharanya02/fake-news-detection/input

+ ### Getting Real about Fake News Dataset :-

The dataset contains text and metadata from 244 websites and represents 12,999 posts.The link for the dataset is https://www.kaggle.com/datasets/mrisdal/fake-news?resource=download

+ ### Global News Dataset :-
This news data set contains various news of global scope . This helps the model to get trained for international news from all domains .The link for the dataset is https://www.kaggle.com/datasets/dbs800/global-news-dataset

+ ### real_or_fake Dataset :-

The dataset contains three coulmns namely **title**,**text** and **label** along with 166356 rows.The link for the dataset is https://www.kaggle.com/datasets/rchitic17/real-or-fake

# Pre-Processing Stage :-

In this stage, the dataset has images, URLs, non-English characters, and missing values that can hinder the process of analyzing the dataset. To remove them techniques such as punctuation removal, stemming, stopword removal and missing values removal have been used to convert it into a simple text dataset so that our classifiers work blissfully. As our data in the datasets were in the form of sentences our models would not be able to read them. To avoid this inconvenience we utilized TF− IDF Vectorizer which converted the sentences into the language that can be interpreted by the machine.

1) **Stopwords Removal:** Stopword removal and stemming
are preprocessing techniques used in Natural Language Processing (NLP) to improve analysis. Stop words are commonly
used words in a language that have little semantic weight and don’t add much information to text. Examples of stop words include articles, prepositions, pronouns, and conjunctions. For example, ”the”, ”a”, ”an”, ”so”, and ”what” are all stopwords in English. In this stage, we have removed them from all of the sentences that have their presence in our datasets.
 
2) **Stemming:** It’s a technique in which there’s a decreasing emphasis on the base-root form of the word. For this case,
 ”ring” could be a root-base word, and ”ringing” and ”rings” are differentiating shapes of that word. Another example is,
 that ”laugh” could be a root-base word, and ”laughs” and ”laughing” are differentiating shapes of that word.
 
3) **Punctuation Removal:** By removing the punctuation marks such as removing of full stops, commas, hyphens, etc. After that, all the text gets converted into space-separated sequences of words.
 
4) **Removing missing values:** In datasets, missing information or lost values emerge when no information esteem is protected for the variable within the perception. The lost information is killed by spaces.

5) **Tokenization:** It is a preprocessing technique in which the data are broken down into single units that are known as ‘tokens’.For example: “How are you” is broken down into three parts “How”, ”are”, and ”you”. Another example is ”The Weather is Good” which is broken down into four parts namely ”The”, ”weather”, ”is”, and ”good”.

# Feature Extraction :-

In this stage, we have utilized TF-IDF for extraction of certain features which are then utilized to evaluate the importance of words in the sentence. TF-IDF is also useful for text classification and helps machine learning models to read words.

# Training the machine learning model :-

Mostly three basic machine learning classifiers were implemented. Those are, Logistic Regression, Decision Tree, and Support Vector Machine algorithms.
