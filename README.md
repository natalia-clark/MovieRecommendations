# Movie Recommendation RAG System 

## Installation Instructions 
Welcome to the Movie Recommendations application! This guide will walk you through the steps to set up and run the application on your local machine.

#### Prerequisites
Before proceeding, ensure you have the following installed:

- Python (version 3.6 or higher)
- Git (optional for cloning the repository)
- Git LFS (for handling large files if not already installed)

#### Installation Steps

1. Clone the Repository (if not downloaded yet):
``` python
git clone https://github.com/natalia-clark/MovieRecommendations.git
cd MovieRecommendations
```
2. Install dependencies:
pip install -r requirements.txt

3a. Set up Git LFS if not configured: https://git-lfs.com/
``` python
git lfs install
git lfs pull
```

3b. Download Large Files from Google Drive (if needed)
If Git LFS isn't functioning or you prefer an alternative method, download the large files from the Google Drive link provided in the repository's README.md at the end.

4. Run the application
``` python
cd MovieRecommendations
streamlit run streamlit.py
```
- Note: If you prefer to look into the training, you can see the file modelB.ipynb. Otherwise, intermediate steps have been saved to be able to run the front-end application with real-time predictions.

#### Additional Notes 
**Data Files:**
- movie_scripts.csv: CSV file containing movie script data.
- movie_scripts_df.pkl: Cleaned movie scripts from preprocessing function 
- features_df.pkl: Extracted fatures from the preprocessed scripts using BERT
- lda_model.pkl: Pickle file containing a trained LDA (Latent Dirichlet Allocation) model.
- dropbox-archive: Folder containing the movie script raw text in a subfolder called movie_scripts

**Scripts/Notebooks/PDFs:**
- modelB.py: Python script for model implementation in streamlit app.
- streamlit.py: Streamlit script to run the application locally.
- modelB.ipynb: Jupyter notebook for model implementation.
- ConciseDescription_NLP.pdf: PDF document providing a concise overview of the NLP (Natural Language Processing) techniques used.

This ReadME.md provides all the information necessary to run this project locally!

---
## Introduction

In this analysis, we will address a RAG problem: How can I get a movie recommendation based on the scripts of the movies?

We see many different potential unique value propositions for this in terms of validating movie content based on the literal content rather than the content contained in reviews. While reviews are valuable, there are systems that analyze reviews and give you recommendations for movies based on reviews. Sometimes, you may not have the same tastes as reviewers or you may not agree with human opinions.

For this, we will apply deep learning models to receive human text explanations of desired movie descriptions to see and recommend movies where their scripts match the themes outlined in the query.

Specifically, this model will employ **Transfer Learning** using a pretrained model on the English language and then supplied with movie scripts. In the future, this project could grow to include more movie scripts, but for the scope of this assignment, we focus our analysis.

Our Basic Strategy:

1. Employ a pre-trained English language model: BERT. This model is able to understand sentiments and such to classify English text.
2. Preprocess the film text to be better prepared to be sentiment analyzed. The preprocessing of the scripts is time-consuming. Here are the basic steps:
  - 2a. Tokenization: Split the script text into sentences and words.
  - 2b. POS Tagging and Lemmatization: Assign parts of speech to each word and reduce them to their base forms.
  - 2c. NER and Theme Extraction: Identify named entities (e.g., persons, organizations, locations) and extract themes.
  - 2d. Stop Words Removal and Filtering: Remove common stop words and non-alphanumeric tokens.
  - 2e. Sentiment Analysis: Analyze the sentiment of each sentence to understand the overall tone of the script.
3. Generate embeddings using BERT for both the preprocessed movie scripts and the user query to capture semantic content.
4. Measure similarity between the user query and the scripts using cosine similarity, considering both semantic content and sentiment.
5. Recommend the most relevant movies based on the combined similarity scores.

From these recommendations, we run our front end application using **streamlit**.

---
## Further Considerations / Limitations
We only trained on a corpus of 1092 movie scripts and made some naive decisions on text processing due to computational power. For example, we limited the sentiment analysis comparison between user queries and movies to be a simple measure of distance, but this relies on the original preprocessing to be rich enough in its analysis to calculate that distance with sophistication. In the future, with more time and computational power, we could dive into that process a bit more to make sure the preprocessing of the scripts was done with advanced NLP techniques.


To be able to download the rest of the intermediate model df and csv, including the original zip for the dataset, please go to the link:
https://drive.google.com/drive/folders/1vYTxei_0bfMzzVEOsdYY6ejcvJ1Cf1V4?usp=sharing
