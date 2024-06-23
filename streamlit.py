import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import pickle
# importing functions already written in modelB.py
from modelB import recommend_movies_with_scores
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
from transformers import BertTokenizer, BertModel
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

# Custom CSS for styling
st.markdown('''
    <style>
    .content {
        color: black;
        padding: 2rem;
    }
    h1 {
        color: #FFD700;
        font-size: 2.5rem;
        text-align: center;
    }
    h2 {
        color: #FF6347;
        font-size: 2rem;
        margin-top: 2rem;
    }
    p {
        font-size: 1.2rem;
        text-align: justify;
    }
    .image-center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 60%;
    }
    .stApp {
        background-image: url("https://www.pngkey.com/png/detail/120-1208201_available-in-two-formats-marco-de-foto-rollo.png");
        background-size: cover;
    }
    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    .content-box {
        position: relative;
        z-index: 1;
        color: white;
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
    }
    </style>
    ''', unsafe_allow_html=True)

# Define the main page
def main_page():
    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    image_url1 = "https://i.ibb.co/c8C2Nbw/Picture1.png"
    st.markdown(f'''
    <div style="text-align: center;">
        <img src="{image_url1}" alt="Welcome to MovieScripted" style="width: 60%; height: auto;">
    </div>
    ''', unsafe_allow_html=True)
        
    st.markdown('''
    # Welcome to MovieScripted
    
    ## Overview
    Welcome to MovieScripted, the innovative movie recommendation system that goes beyond reviews and ratings. At MovieScripted, we believe that the heart of a movie lies in its script. By analyzing the scripts of movies, we can provide you with personalized recommendations that perfectly match your mood and preferences.

    ## The Problem
    Traditional movie recommendation systems often rely heavily on user reviews, ratings, and viewing history to suggest films. While this method has its merits, it overlooks a crucial aspect—the actual content of the movie's script. Reviews and ratings can be biased or influenced by factors unrelated to the movie's core themes and emotions. As a result, finding movies that truly resonate with your emotional state or specific interests can be challenging.

    ## Our Solution
    MovieScripted addresses this gap by focusing on the scripts themselves. Using advanced Natural Language Processing (NLP) techniques, we analyze the content of movie scripts to understand their underlying emotions, themes, and narrative styles. Here’s how we do it:

    ### * Script Analysis
    We collect and preprocess a vast dataset of movie scripts, cleaning and tokenizing the text to prepare it for analysis.
    ''')

    # Add the second picture
    image_url2 = "https://media.licdn.com/dms/image/C4E12AQHkxT3iXbNrrQ/article-cover_image-shrink_720_1280/0/1551990908321?e=2147483647&v=beta&t=5zNDuJ24pDE_d9n3mytDUCFEMKL1XDl65CPjrX3zLyE"
    st.markdown(f'''
    <div style="text-align: center;">
        <img src="{image_url2}" alt="Welcome to MovieScripted" style="width: 60%; height: auto;">
    </div>
    ''', unsafe_allow_html=True)
        
    st.markdown('''
    ### * Sentiment and Emotion Detection
    Our system uses sentiment analysis to gauge the overall mood of each script and tokenizes the text to be able to be read by a BERT model.

    ### * Topic Modeling
    We employ Latent Dirichlet Allocation (LDA) to extract key topics and themes from the scripts, allowing us to categorize movies based on their content.
                
    ### * Personalized Recommendations
    Based on your input—whether you're looking for a happy movie, an action-packed adventure, or a thought-provoking drama—our recommendation algorithm matches your preferences with the most suitable scripts.

    ## How It Works
    ### 1 - Input Your Preferences
    Let us know what kind of movie you’re in the mood for by selecting emotions, themes, or specific keywords.

    ### 2 - Script Analysis
    Our NLP models has already processed the scripts in our database to find movies that align with your preferences.

    ### 3 - Get Recommendations
    Receive a curated list of movies to help you decide your next watch.

    ## Explore and Discover
    Dive into the world of movies like never before with MovieScripted. Explore our recommendations, discover hidden gems, and enjoy films that truly speak to your heart.

    ## Welcome to the future of movie recommendations...
    ## Welcome to MovieScripted!!!
    ''')

    # Add the third picture
    image_url3 = "https://media.istockphoto.com/id/1753952419/photo/man-using-mouse-and-keyboard-for-streaming-online-watching-video-on-internet-show-or-tutorial.jpg?b=1&s=170667a&w=0&k=20&c=Y8C43NWC5aHRzW6GA3CYDPnSiv2Sj_Q3mWWMsvSLU4c="
    st.markdown(f'<img src="{image_url3}" class="image-center" alt="Welcome to MovieScripted" style="width: 60%; height: auto; display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def read_movie_scripts():
    # Load the movie scripts dataframe
    movie_scripts_df = pd.read_pickle('movie_scripts_df.pkl')
    titles = movie_scripts_df['Title']
    preprocessed_scripts = movie_scripts_df['Preprocessed Script']
    themes = movie_scripts_df['Themes']
    sentiments = movie_scripts_df['Sentiments']
    lda = pickle.load(open('lda_model.pkl', 'rb'))

    features_df = pd.read_pickle('features_df.pkl')
    features = features_df.values

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    return titles, preprocessed_scripts, themes, sentiments, lda, features, tokenizer, model

titles, preprocessed_scripts, themes, sentiments, lda, features, tokenizer, model = read_movie_scripts()

# Define the movie recommendation page
def movie_recommendation_page():
    
    # Add a background image using CSS and HTML
    page_bg_img = '''
    <style>
    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    .content-box {
        position: relative;
        z-index: 1;
        color: white;
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
    }
    h1, h2 {
        color: #FFD700;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Add a div for the overlay
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

    st.title("Movie Recommendation System")

    st.write("### Enter movie preferences:")
    movie_input = st.text_input("I want a movie that has...", "fashion, dogs, and scary")

    if st.button("Get Recommendation"):

        recommended_movies = recommend_movies_with_scores(movie_input, tokenizer, model, features, titles, sentiments)
        movie_recs = [movie[0] for movie in recommended_movies]
        # movie_image_url = "https://prod-ripcut-delivery.disney-plus.net/v1/variant/disney/15A4BC0BAD7442F99A2F6C2CE66B99C99E535BF9690A256DAB29C6EDF1B04866/scale?width=1200&aspectRatio=1.78&format=webp"  # Replace with actual image URL
        st.write(f"### Recommended Movies:\n")
        for title in movie_recs:
            st.write(f"{title}")
        # ending white box behind 
    st.markdown('</div>', unsafe_allow_html=True)

def page_2():
    st.markdown('''
    <style>
    .content {
        color: black;
        padding: 2rem;
    }
    h1 {
        color: #FFD700;
        font-size: 2.5rem;
        text-align: center;
    }
    h2 {
        color: #FF6347;
        font-size: 2rem;
        margin-top: 2rem;
    }
    p {
        font-size: 1.2rem;
        text-align: justify;
    }
    .image-center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 60%;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown('''
    # Understanding the Problem

    ## The Challenge with Traditional Recommendation Systems

    Traditional movie recommendation systems rely heavily on user-generated content, such as:

    - **Ratings**: Scores given by viewers based on their overall enjoyment.
    - **Reviews**: Written feedback that often includes personal opinions, experiences, and biases.
    - **Viewing History**: Data on what users have watched before, which is used to predict future preferences.

    While these methods can be effective, they come with significant limitations:

    - **Bias and Subjectivity**: Reviews and ratings can be highly subjective and influenced by individual tastes, social trends, and external factors unrelated to the movie's core content.
    - **Popularity Over Quality**: Popular movies with high ratings often dominate recommendations, overshadowing lesser-known but potentially more suitable films.
    - **Surface-Level Insights**: Traditional methods provide a surface-level understanding of a movie, missing deeper narrative elements and emotional undertones embedded in the script.
    ''')

    image_url1 = "https://www.nzherald.co.nz/resizer/Cq4YHsbTEHQXosZsE1zf5pqHk6A=/1200x675/smart/filters:quality(70)/cloudfront-ap-southeast-2.images.arcpublishing.com/nzme/FBSVI36QISB7C57ZSE5BVROGG4.jpg"
    st.markdown(f'''
    <div style="text-align: center;">
        <img src="{image_url1}" alt="Welcome to MovieScripted" style="width: 60%; height: auto;">
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    ## Why This Matters

    When seeking a specific type of movie—such as one that will uplift your spirits or make you ponder deeply—the traditional approach may not deliver the best matches. You might end up watching movies that are popular or highly rated but do not align with the emotional or thematic experience you are looking for.

    ## How MovieScripted Addresses the Problem

    ### Script-Centric Analysis

    MovieScripted shifts the focus from user opinions to the actual content of the movie scripts. Here's how our approach provides a more nuanced and accurate recommendation:

    - **Deep Text Analysis**: By analyzing the full text of movie scripts, we gain insights into the story's core elements, such as tone, themes, and emotional arcs.
    - **Sentiment and Emotion Detection**: We employ advanced NLP techniques to detect the overall sentiment and specific emotions conveyed in the script. This helps us understand whether a movie is uplifting, melancholic, suspenseful, etc.
    - **Keyword and Theme Extraction**: Our system extracts key themes and concepts from the script, ensuring that the recommended movies match the user's desired topics or interests.

    ### Our Solution in Action

    - **User Input**: You tell us what you're in the mood for—be it a happy, exciting, or thought-provoking movie.
    - **NLP Processing**: Our algorithms analyze a comprehensive database of movie scripts, identifying those that best match your specified emotions and themes.
    - **Personalized Recommendations**: We provide you with a list of movie recommendations that align closely with your input, complete with script excerpts and summaries to help you choose.

    ## Why MovieScripted is Worth It

    ### Personalized Experience

    MovieScripted offers a highly personalized movie-watching experience. By focusing on the script, we ensure that the movies you watch resonate with your current mood and interests, leading to a more fulfilling viewing experience.

    ### Discover Hidden Gems

    Our system brings to light movies that might not be mainstream but have rich narratives and emotional depth. This helps you discover hidden gems that traditional recommendation systems might overlook.

    ### Informed Choices

    With detailed insights into the script's content, you make informed decisions about what to watch. You know not just the genre and rating, but the emotional and thematic journey the movie offers.

    ### Enhanced Satisfaction

    By aligning movie recommendations with your specific emotional and thematic preferences, MovieScripted enhances your overall satisfaction with your movie choices, making each viewing experience more meaningful.
    ''')

    st.markdown('</div>', unsafe_allow_html=True)

def page_4():
    st.markdown('''
    <style>
    .content {
        color: black;
        padding: 2rem;
    }
    h1 {
        color: #FFD700;
        font-size: 3rem;
        text-align: center;
    }
    h2 {
        color: #FFD700;
        font-size: 2rem;
        margin-top: 2rem;
    }
    p {
        font-size: 1.2rem;
        text-align: justify;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    st.markdown('<h1>Future Improvements</h1>', unsafe_allow_html=True)
    
    st.markdown('''
    ## Expanding Our Movie Database
    To provide even more accurate and diverse recommendations, one of our primary goals is to expand our movie script database. By including more scripts from a wide range of genres, languages, and time periods, we can:
    - **Broaden Recommendations:** Offer a wider selection of movies that cater to diverse tastes and preferences.
    - **Enhance Accuracy:** Improve the precision of our recommendations by having a richer pool of scripts to analyze.
    - **Cultural Diversity:** Include films from different cultures and countries, allowing users to explore global cinema.
    
    ## Advanced NLP Techniques
    We aim to continuously enhance our NLP models to understand scripts better and provide even more refined recommendations:
    - **Emotion Detection:** Develop more sophisticated emotion detection algorithms that can distinguish between subtle emotional nuances in scripts.
    - **Contextual Analysis:** Implement advanced contextual analysis to better understand the interplay between different scenes, characters, and dialogues.
    - **Theme Clustering:** Improve theme extraction and clustering techniques to identify complex and overlapping themes within scripts.
     
    ## Enhanced User Interface
    We plan to enhance the user interface to make the experience even more engaging and user-friendly:
    - **Interactive Visualizations:** Implement interactive visualizations that allow users to explore the emotional and thematic landscape of recommended movies.
    - **Detailed Summaries:** Provide more detailed summaries and script excerpts to help users make informed choices.
    - **User Feedback Integration:** Integrate user feedback mechanisms to continually improve the recommendation system based on user experiences.
    
    ## Integration with Streaming Services
    To make it easier for users to watch recommended movies, we aim to integrate our system with popular streaming services:
    - **Direct Links:** Provide direct links to movies on streaming platforms, making it seamless for users to watch recommended films.
    - **Availability Notifications:** Notify users when a recommended movie becomes available on their preferred streaming service.
    
    ## Community and Social Features
    Building a community around MovieScripted can enhance user engagement and satisfaction:
    - **User Reviews and Ratings:** Allow users to rate and review movies based on their script content, adding another layer of personalized recommendations.
    - **Discussion Forums:** Create discussion forums where users can share their thoughts and recommendations, fostering a sense of community.
    - **Social Sharing:** Enable users to share their favorite movie recommendations with friends and family on social media.
    
    ## Continuous Improvement and Feedback Loop
    We are committed to continuous improvement based on user feedback and technological advancements:
    - **Research and Development:** Invest in ongoing research and development to stay at the forefront of NLP and recommendation technologies.
    - **Regular Updates:** Provide regular updates to the system, ensuring it remains relevant and effective in meeting user needs.
    
    ## Achieving Our Vision
    With these future improvements, MovieScripted aims to become the ultimate destination for personalized movie recommendations based on script analysis. By continually expanding our database, enhancing our technology, and focusing on user satisfaction, we strive to create an unparalleled movie discovery experience that resonates deeply with each user’s unique preferences.
    ''')

    st.markdown('</div>', unsafe_allow_html=True)

def our_methodology_page():
    st.title("Our Methodology")
    # read txt from first script in dropbox-archive/movie_scripts folder 
    script_path = "dropbox-archive/movie_scripts/Script_Lost Horizon.txt"
    with open(script_path, 'r') as file:
         # read first script 
        script = file.read()
    preprocessed_script1 = preprocessed_scripts[0]
    
    st.write(f"### Example Original Script:")
    st.markdown(f'''
    <div style="text-align: center;">
    </div>
    ''', unsafe_allow_html=True)
    st.markdown(script[:1000]) 
    st.write(f"### Example Preprocessed Script:")
    st.markdown(f'''
    <div style="text-align: center;">
    </div>
    ''', unsafe_allow_html=True)
    st.markdown(preprocessed_script1[:1000])
    # add a horizontal line 
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'''
    <div style="text-align: center;">
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('''
    ### Preprocessing Stage
    
    This function performs the following steps:
    1. Tokenization - Splitting the script into words
    2. Stopword removal - Removing common words like the, is, and and
    3. Lemmatization - Converting words to their base form
    4. Sentence segmentation - Splitting the script into sentences
    5. Named Entity Recognition (NER) - Extracting entities like PERSON, ORGANIZATION, GPE
    6. Sentiment analysis - Using TextBlob to calculate sentiment polarity
    7. Topic modeling - Using Latent Dirichlet Allocation (LDA) to extract topics - Optional
    
    It returns the preprocessed scripts, themes, sentiments, and the LDA model.
    
    The LDA model can be used for topic-based recommendation.
    ''')
    # add code block to include the preprocessing function 
    st.markdown('''
    ```python
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
```''')
     # add a horizontal line 
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'''
    <div style="text-align: center;">
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('''
    ### Feature Extraction Stage
                
    The next step we take, after preprocessing the scripts, is to extract features from the preprocessed scripts. We use the BERT model to encode the preprocessed scripts into dense vectors that capture the semantic meaning of the text. These vectors are then used as features for similarity calculations.
    ''')
    # add code block to include the feature extraction function
    st.markdown('''
    ```python
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
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
    ''')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'''
    <div style="text-align: center;">
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('''
    ### Recommendation Stage (RAG)
    
    The last step we take is to recommend movies by accepting user input queries, encoding them using the BERT model, and calculating the cosine similarity between the user query and the preprocessed scripts. The movies with the highest similarity scores are recommended to the user. This is known as a RAG system.

    **What is a RAG system?:** A RAG (Retrieval-Augmented Generation) system combines information retrieval with natural language generation to provide relevant and contextually appropriate responses or recommendations based on user inputs.

    The recommendation is based on the following steps:
    1. **Extract features for the user input** using BERT, similar to how we extracted features from the scripts
        - BERT (Bidirectional Encoder Representations from Transformers) is utilized to convert the user input text into dense, context-aware embeddings that capture its semantic meaning.
    2. Perform **sentiment analysis on the user input** using TextBlob
        - TextBlob is employed here to analyze the sentiment polarity of the user input text. This step helps in understanding the emotional tone conveyed by the user.
    3. Calculate **cosine similarity between user input and movie features**
        - Cosine similarity measures the cosine of the angle between two vectors and is used here to assess the similarity between the user input features and the features extracted from movie scripts.
    4. Calculate **sentiment differences between user input and scripts**
        - Sentiment differences are computed by comparing the sentiment polarity of the user input with that of each movie script. This step quantifies the emotional distance between the user input and each script.
    5. Combine similarity scores and sentiment differences for **final scoring of similarities** between user input and the scripts
        - The combined scores integrate the cosine similarity scores and sentiment differences to produce a holistic measure of similarity between the user input and the movie scripts.
    6. Get the **top 5 recommended movies** based on combined scores and return them to the user

    ''')
    st.markdown('''
            ```python
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
                ''')

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar navigation with collapsible sections
st.sidebar.title("MovieScripted")
page = st.sidebar.selectbox("Navigation", ["Get to know Moviescripted", "Why and How", "Try Our Demo Now!", "Future Development", "Our Methodology"])

# Page display logic
if page == "Get to know Moviescripted":
    main_page()
elif page == "Why and How":
    page_2()
elif page == "Try Our Demo Now!":
    movie_recommendation_page()
elif page == "Future Development":
    page_4()
elif page == "Our Methodology":
    our_methodology_page()
