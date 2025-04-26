# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


'''
This project is designed to test your understanding of how to build a high-performing classification algorithm to predict 
whether a particular movie on Rotten Tomatoes is labeled as 'Rotten', 'Fresh', or 'Certified-Fresh'.

There are two approaches that we're going to learn to predict a movie's status in this data project:

by using numerical and categorical features
by using text data (review from the critics)

The dataset that we're going to use to apply each of the approaches above will be different.
 For the first approach, we're going to use rotten_tomatoes_movies.csv as our dataset. 
 For the second approach, we're going to use rotten_tomatoes_critic_reviews_50k.csv

You can build a machine learning model with any algorithm, but in this task, we're going to focus our attention on predicting 
the movie status with tree-based algorithms, i.e Decision Tree Classifier and Random Forest algorithm. If you want to learn about the 
in-depth explanation of the learning process of the Decision Tree and Random Forest algorithm, you can read more about it here.
'''

#First Approach: Predicting Movie Status Based on Numerical and Categorical Features

# Read movie data
df_movie = pd.read_csv('rotten_tomatoes_movies.csv')
print(df_movie.head())

# Check data distribution
print(df_movie.describe())

#Data Preprocessing
print(f'Content Rating category: {df_movie.content_rating.unique()}')

# Visualize the distribution of each category in content_rating feature
ax = df_movie.content_rating.value_counts().plot(kind='bar', figsize=(12,9))
ax.bar_label(ax.containers[0])

# One hot encoding content_rating feature
content_rating = pd.get_dummies(df_movie.content_rating)
print(content_rating.head())
