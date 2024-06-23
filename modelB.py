#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this analysis, we will address a RAG problem: How can I get a movie recommendation based on the scripts of the movies?
# 
# We see many different potential unique value propositions for this in terms of validating movie content based on the literal content rather than the content contained in reviews. While reviews are valuable, there are systems that analyze reviews and give you recommendations for movies based on reviews. Sometimes, you may not have the same tastes as reviewers or you may not agree with human opinions.
# 
# For this, we will apply deep learning models to receive human text explanations of desired movie descriptions to see and recommend movies where their scripts match the themes outlined in the query.
# 
# Specifically, this model will employ **Transfer Learning** using a pretrained model on the English language and then supplied with movie scripts. In the future, this project could grow to include more movie scripts, but for the scope of this assignment, we focus our analysis.
# 
# Our Basic Strategy:
# 
# 1. Employ a pre-trained English language model: BERT. This model is able to understand sentiments and such to classify English text.
# 2. Preprocess the film text to be better prepared to be sentiment analyzed. The preprocessing of the scripts is time-consuming. Here are the basic steps:
# - 2a. Tokenization: Split the script text into sentences and words.
# - 2b. POS Tagging and Lemmatization: Assign parts of speech to each word and reduce them to their base forms.
# - 2c. NER and Theme Extraction: Identify named entities (e.g., persons, organizations, locations) and extract themes.
# - 2d. Stop Words Removal and Filtering: Remove common stop words and non-alphanumeric tokens.
# - 2e. Sentiment Analysis: Analyze the sentiment of each sentence to understand the overall tone of the script.
# 3. Generate embeddings using BERT for both the preprocessed movie scripts and the user query to capture semantic content.
# 4. Measure similarity between the user query and the scripts using cosine similarity, considering both semantic content and sentiment.
# 5. Recommend the most relevant movies based on the combined similarity scores.
# 
# From these recommendations, we run our front end application using **streamlit**.
# 

# In[19]:


import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, pipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from textblob import TextBlob
import gensim
from gensim import corpora


# In[20]:


# Ensure you have the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


# ## Step 1: Load the Data
# This code represents a function to load the scripts from the directory of scripts and preprocess the naming conventions based on how the scripts are currently named. 
# 
# Reference for the script downloads: https://osf.io/zytmp/files/dropbox
# 
# Description of the dataset:
# 
# > "This dataset contains 1,093 movie scripts collected from the website imsdb.com, each in a separate text file. The file imsdb_sample.txt contains the titles of all movies (corresponding file names are in the form Script_TITLE.txt) The website was crawled in January 2017. Some scripts are not present as they were missing in imsdb.com or because they were uploaded as pdf files. Please notice that (i) the original scripts were uploaded on the website by individual users, so that they might not correspond exactly to the movie scripts and typos may be present; (ii) html formatting was not consistent in the website, and so neither is the formatting of the resulting text files. Even considering (i) and (ii), the quality seems good on average and the dataset can be easily used for text-mining tasks."

# In[21]:


# Function to read scripts from files and extract titles
def load_scripts_from_directory(directory_path):
    scripts = []
    titles = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                scripts.append(content)
                # Extract title from the filename (remove "Script_" and ".txt")
                title = filename.replace("Script_", "").replace(".txt", "").replace("_", " ")
                titles.append(title)
    return titles, scripts


# ## Step 2: Preprocess the Scripts
# 
# This function deals with the preprocessing tasks for the scripts. Before using BERT to get the embeddings of the scripts, we need to properly preprocess the scripts. Here is the approach:
# 1. **Tokenization:** The process of splitting text into smaller units called tokens. In this case, we split the script into sentences and then each sentence into words.
# 2. **Stopword Removal**: Stopwords are common words like "the", "is", and "and" that do not contribute significant meaning to the text. Removing these words helps to focus on the more meaningful parts of the text.
# 3. **Lemmatization**: The process of converting words to their base form. For example, "running" becomes "run". This helps to reduce the variability of words and focus on their core meaning.
# 4. **Sentence Segmentation**: The process of splitting the script into individual sentences. This helps in analyzing the script at a more granular level.
# 5. **Named Entity Recognition (NER)**: The process of identifying named entities in the text, such as persons, organizations, and geopolitical entities. This helps in understanding the key entities in the script.
# 6. **Sentiment Analysis**: The process of determining the sentiment or emotional tone of the text. In this case, we use TextBlob to calculate the sentiment polarity of each sentence. TextBlob evaluates the sentiment polarity of a piece of text, indicating whether the expressed opinion is positive, negative, or neutral.
# 7. **Topic Modeling**: The process of identifying the topics present in a collection of documents. We use Latent Dirichlet Allocation (LDA) to extract topics from the preprocessed scripts. 
#     - To explain LDA, consider this example for our collection of movie scripts: Using LDA, you could identify topics such as "romance", "action", "comedy", etc. Each script would have a distribution over these topics, indicating the prominence of each theme in that script. Moreover, each topic would have a distribution of words that are likely to occur in scripts belonging to that topic, providing insight into the thematic content of the scripts.

# In[22]:


'''
Function to preprocess movie scripts using basic text processing techniques.
This function performs the following steps:
1. Tokenization - Splitting the script into words
2. Stopword removal - Removing common words like "the", "is", "and"
3. Lemmatization - Converting words to their base form
4. Sentence segmentation - Splitting the script into sentences
5. Named Entity Recognition (NER) - Extracting entities like PERSON, ORGANIZATION, GPE
6. Sentiment analysis - Using TextBlob to calculate sentiment polarity
7. Topic modeling - Using Latent Dirichlet Allocation (LDA) to extract topics - Optional
It returns the preprocessed scripts, themes, sentiments, and the LDA model.
The LDA model can be used for topic-based recommendation.
'''
def preprocess_scripts_advanced(scripts):
    # setting up the stop words
    stop_words = set(stopwords.words('english'))
    # setting up the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Initialize lists to store preprocessed scripts, themes, sentiments
    preprocessed_scripts = []
    themes = []
    sentiments = []
    
    # Process each script
    for script in scripts:
        # Tokenize script into sentences
        sentences = sent_tokenize(script.lower())
        filtered_script = []

        # Process each sentence
        for sentence in sentences:
            # Tokenize sentence into words
            doc = nlp(sentence)
            # Extract entities using NER
            word_tokens = word_tokenize(sentence)

            # Initialize list to store filtered words
            filtered_sentence = []
            sentence_themes = []
            
            # Lemmatize words based on POS tags
            for word, pos in pos_tag(word_tokens):
                # Check if the word is an alphanumeric and not in stop words
                if word.isalnum() and word.lower() not in stop_words:
        
                    if pos.startswith('NN'):  # Noun
                        filtered_sentence.append(lemmatizer.lemmatize(word, pos='n'))
                    elif pos.startswith('VB'):  # Verb
                        filtered_sentence.append(lemmatizer.lemmatize(word, pos='v'))
                    elif pos.startswith('JJ'):  # Adjective
                        filtered_sentence.append(lemmatizer.lemmatize(word, pos='a'))
                    elif pos.startswith('RB'):  # Adverb
                        filtered_sentence.append(lemmatizer.lemmatize(word, pos='r'))
                    else:
                        filtered_sentence.append(word)
            
            # Append sentence for context
            filtered_script.append(' '.join(filtered_sentence))
            
            # Extract themes using NER and keyword matching
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                    sentence_themes.append(ent.label_.lower())
            
            themes.append(' '.join(sentence_themes))
            
            # Sentiment analysis
            blob = TextBlob(sentence)
            # Append sentiment polarity
            sentiments.append(blob.sentiment.polarity)
        
        # Append preprocessed script
        preprocessed_scripts.append(' '.join(filtered_script))
    
    # Topic modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # Fit and transform the preprocessed scripts
    dtm = vectorizer.fit_transform(preprocessed_scripts)
    # Fit LDA model which will extract 20 topics
    lda = LatentDirichletAllocation(n_components=20, random_state=42)
    # Fit the model
    lda.fit(dtm)
    
    # Return preprocessed scripts, themes, sentiments, and LDA model
    return preprocessed_scripts, themes, sentiments, lda


# In[23]:


# Path to the directory containing the script files
# directory_path = "dropbox-archive/movie_scripts/"

# # Load scripts and titles
# titles, scripts = load_scripts_from_directory(directory_path)

# # Preprocess the scripts
# preprocessed_scripts, themes, sentiments, lda = preprocess_scripts_advanced(scripts)


# We are going to save the preprocessed scripts and the LDA model to not have to run this long execution again.

# In[35]:


# # save LDA model
# pd.to_pickle(lda, 'lda_model.pkl')


# # In[38]:


# print(type(titles), len(titles))
# print(type(preprocessed_scripts), len(preprocessed_scripts))
# print(type(themes), len(themes))
# print(type(sentiments), len(sentiments))


# This code below is for the themes and sentiments to be shortened to be script wide, not sentence wide. I had to reparse through, but for the sake of not running the long preprocessing function again, we did it post-processing to be able to save the results to a df.

# In[43]:


# # List to store the aggregated sentiments and themes per script
# aggregated_sentiments = []
# aggregated_themes = []

# # Placeholder variables to track per script processing
# current_script_sentiments = []
# current_script_themes = []

# # Track the current script index
# script_index = 0
# sentence_count = len(sent_tokenize(scripts[script_index].lower()))

# for i in range(len(sentiments)):
#     current_script_sentiments.append(sentiments[i])
#     current_script_themes.extend(themes[i].split())
    
#     if i + 1 >= sentence_count:  # Move to the next script
#         # Average the sentiments for the current script
#         aggregated_sentiments.append(np.mean(current_script_sentiments) if current_script_sentiments else 0)
#         # Collect the themes for the current script
#         aggregated_themes.append(current_script_themes)
        
#         # Reset placeholder variables for the next script
#         current_script_sentiments = []
#         current_script_themes = []
        
#         # Increment the script index and calculate the new sentence count
#         script_index += 1
#         if script_index < len(scripts):
#             sentence_count += len(sent_tokenize(scripts[script_index].lower()))

# # For any leftover data
# if current_script_sentiments:
#     aggregated_sentiments.append(np.mean(current_script_sentiments) if current_script_sentiments else 0)
#     aggregated_themes.append(current_script_themes)


# # In[52]:


# # fixing the aggregated themes into one value which is the mode of the themes
# aggregated_themes = [max(set(theme), key=theme.count) for theme in aggregated_themes]


# # In[55]:


# # Now aggregated_sentiments and aggregated_themes should have the correct sizes matching the number of scripts
# print(len(aggregated_sentiments))  # Should match the number of scripts
# print(len(aggregated_themes))  # Should match the number of scripts
# print(len(titles))  # Should match the number of scripts    
# print(len(preprocessed_scripts))  # Should match the number of scripts


# In[95]:


# # constructing dataframe
# df = pd.DataFrame({
#     'Title': titles, 
#     'Preprocessed Script': preprocessed_scripts, 
#     'Themes': aggregated_themes, 
#     'Sentiments': aggregated_sentiments})
# df.columns = df.columns.astype(str)
# # fixing types to be strings, not object 
# df['Title'] = df['Title'].astype(str)
# df['Preprocessed Script'] = df['Preprocessed Script'].astype(str)
# df['Themes'] = df['Themes'].astype(str)
# df['Sentiments'] = df['Sentiments'].astype(float)
# df = df.reset_index(drop=True)
# df.head()


# In[85]:


# # save the dataframe
# df.to_pickle('movie_scripts_df.pkl')


# Reading in the saved files again.

# In[87]:


# # reading in pre-saved preprocessed data
# df = pd.read_pickle('movie_scripts_df.pkl')

# # Extracting the titles, preprocessed scripts, themes, and sentiments
# titles = df['Title']
# preprocessed_scripts = df['Preprocessed Script']
# themes = df['Themes']
# sentiments = df['Sentiments']

# # Load the LDA model
# lda = pd.read_pickle('lda_model.pkl')


# In[24]:


# # add comparison of the original and preprocessed script for one example 
# print("Original Script:\n")
# print(scripts[0])
# print("\n\nPreprocessed Script:\n")
# print(preprocessed_scripts[0])
# print ("\n\nThemes:\n", themes[0])
# print ("\n\nSentiment:", sentiments[0])


# # ## Step 3: Importing BERT (Pre-trained model)
# 
# In this section, we will discuss the importation of BERT (Bidirectional Encoder Representations from Transformers), the tokenizer, and the model, followed by their application in the feature extraction process.
# 
# BERT Model and Tokenizer
# BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model developed by Google AI for natural language processing tasks. It has been pre-trained on large amounts of text data and can be fine-tuned for specific tasks such as text classification, question answering, and more.
# - **Tokenizer:** The BertTokenizer tokenizes input text into tokens that BERT understands. It breaks down sentences into subwords (WordPiece tokenizer) and converts these subwords into numerical tokens suitable for BERT input.
# - **Model:** The BertModel is the actual BERT architecture that processes tokenized input. It transforms tokens into contextualized embeddings, capturing the relationships between words in a sentence.

# In[25]:


# # Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')


# #### Extracting Features using BERT
# 
# After importing BERT and the tokenizer/model, we can proceed to extract features.
# 
# - **Feature Extraction:** The extract_features function tokenizes each script using the BERT tokenizer, converts the tokens into tensors, and passes them through the BERT model. The model's outputs include embeddings (vectors representing each token's contextual meaning), which are averaged across all tokens to produce a fixed-size feature vector for each script.
# 
# - **Tensor Handling:** BERT expects inputs as tensors ('pt' for PyTorch tensors in this case), and the outputs are also tensors. These tensors are then converted to NumPy arrays (embeddings) for further processing or analysis.

# In[26]:


'''
Function to extract features from preprocessed scripts using BERT.
This function takes a list of preprocessed scripts, a BERT tokenizer, and a BERT model.
It returns a list of features extracted from the scripts using BERT.
This means that each script will be represented as a 768-dimensional vector.
The vector contains the contextual information of the script based on the BERT model.
'''
def extract_features(scripts, tokenizer, model):
    features = []
    # Process each script
    for script in scripts:
        # Tokenize the script and convert to PyTorch tensors
        inputs = tokenizer(script, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        # Get BERT model outputs and extract the hidden states (last_hidden_state)
        with torch.no_grad(): # Disable gradient calculation, reduces memory consumption
            outputs = model(**inputs)
        # Calculate the mean of the last_hidden_state
        features.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return features

# Extract features from preprocessed scripts
# features = extract_features(preprocessed_scripts, tokenizer, model)


# In[97]:


# save the features to a csv file
# features_df = pd.DataFrame(features)
# # save as pickle 
# features_df.to_pickle('features_df.pkl')


# ## Step 4: Recommendations Based on User Inputs
# 
# The last step of this analysis is the use of a RAG system.
# 
# What is a RAG system?: A RAG (Retrieval-Augmented Generation) system combines information retrieval with natural language generation to provide relevant and contextually appropriate responses or recommendations based on user inputs.
# 
# The recommendation is based on the following steps:
# 1. **Extract features for the user input** using BERT, similar to how we extracted features from the scripts
#     - BERT (Bidirectional Encoder Representations from Transformers) is utilized to convert the user input text into dense, context-aware embeddings that capture its semantic meaning.
# 2. Perform **sentiment analysis on the user input** using TextBlob
#     - TextBlob is employed here to analyze the sentiment polarity of the user input text. This step helps in understanding the emotional tone conveyed by the user.
# 3. Calculate **cosine similarity between user input and movie features**
#     - Cosine similarity measures the cosine of the angle between two vectors and is used here to assess the similarity between the user input features and the features extracted from movie scripts.
# 4. Calculate **sentiment differences between user input and scripts**
#     - Sentiment differences are computed by comparing the sentiment polarity of the user input with that of each movie script. This step quantifies the emotional distance between the user input and each script.
# 5. Combine similarity scores and sentiment differences for **final scoring of similarities** between user input and the scripts
#     - The combined scores integrate the cosine similarity scores and sentiment differences to produce a holistic measure of similarity between the user input and the movie scripts.
# 6. Get the **top 5 recommended movies** based on combined scores and return them to the user

# In[90]:


'''
Function to recommend movies based on user input.
This function takes a user input, a BERT tokenizer, a BERT model, features of preprocessed scripts,
titles of movies, and sentiment scores of scripts.
It returns the top 5 recommended movies based on the user input.   
'''
def recommend_movies_with_scores(user_input, tokenizer, model, features, titles, script_sentiments):
    # Extract features for the user query
    user_input_features = extract_features([user_input], tokenizer, model)[0]
    
    # Perform sentiment analysis on the user query
    user_sentences = sent_tokenize(user_input)
    user_sentiments = [TextBlob(sentence).sentiment.polarity for sentence in user_sentences]
    user_input_sentiment = sum(user_sentiments) / len(user_sentiments) if user_sentiments else 0
    
    # Calculate cosine similarity between user input and movie features
    similarities = cosine_similarity([user_input_features], features)
    
    # Calculate sentiment differences between user input and scripts
    sentiment_differences = [abs(user_input_sentiment - sentiment) for sentiment in script_sentiments]
    
    # Combine similarity scores and sentiment differences for final scoring
    combined_scores = similarities[0] - np.array(sentiment_differences)
    
    # Get the top 5 recommended movies based on combined scores
    recommended_indices = combined_scores.argsort()[-5:][::-1]
    recommended_movies = [(titles[i], combined_scores[i]) for i in recommended_indices]
    
    return recommended_movies


# In[28]:


def input_to_recs(input):
    recommended_movies = recommend_movies_with_scores(input, tokenizer, model, features, titles, sentiments)
    return recommended_movies


# In[99]:


# Example user input for movie description
# user_input = "funny movie"
# recommended_movies_scores = input_to_recs(user_input)
# # extracting just the titles 
# recommended_movies = [movie[0] for movie in recommended_movies_scores]

# print("Query:", user_input, "\nRecommended Movies:",recommended_movies)


# In[94]:


# making requirements.txt file for this project 
# !pip freeze > requirements.txt

