# Movie Recommendation RAG System 

## Introduction

In this analysis, we will address a RAG problem: How can I get a movie recommendation based on the scripts of the movies?

We see many different potential unique value propositions for this in terms of validating movie content based on the literal content rather than the content contained in reviews. While reviews are valuable, there are systems that analyze reviews and give you recommendations for movies based on reviews. Sometimes, you may not have the same tastes as reviewers or you may not agree with human opinions.

For this, we will apply deep learning models to receive human text explanations of desired movie descriptions to see and recommend movies where their scripts match the themes outlined in the query.

Specifically, this model will employ **Transfer Learning** using a pretrained model on the English language and then supplied with movie scripts. In the future, this project could grow to include more movie scripts, but for the scope of this assignment, we focus our analysis.

Our Basic Strategy:

1. Employ a pre-trained English language model: BERT. This model is able to understand sentiments and such to classify English text.
2. Preprocess the film text to be better prepared to be sentiment analyzed. The preprocessing of the scripts is time-consuming. Here are the basic steps:
  2a. Tokenization: Split the script text into sentences and words.
  2b. POS Tagging and Lemmatization: Assign parts of speech to each word and reduce them to their base forms.
  2c. NER and Theme Extraction: Identify named entities (e.g., persons, organizations, locations) and extract themes.
  2d. Stop Words Removal and Filtering: Remove common stop words and non-alphanumeric tokens.
  2e. Sentiment Analysis: Analyze the sentiment of each sentence to understand the overall tone of the script.
3. Generate embeddings using BERT for both the preprocessed movie scripts and the user query to capture semantic content.
4. Measure similarity between the user query and the scripts using cosine similarity, considering both semantic content and sentiment.
5. Recommend the most relevant movies based on the combined similarity scores.

From these recommendations, we run our front end application using **streamlit**.

---
## Further Considerations / Limitations
We only trained on a corpus of 1092 movie scripts and made some naive decisions on text processing due to computational power. For example, we limited the sentiment analysis comparison between user queries and movies to be a simple measure of distance, but this relies on the original preprocessing to be rich enough in its analysis to calculate that distance with sophistication. In the future, with more time and computational power, we could dive into that process a bit more to make sure the preprocessing of the scripts was done with advanced NLP techniques.


To be able to download the rest of the intermediate model df and csv, please go to the link:
https://drive.google.com/drive/folders/1vYTxei_0bfMzzVEOsdYY6ejcvJ1Cf1V4?usp=sharing
