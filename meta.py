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
content_rating.replace({False: 0, True: 1}, inplace=True)
print(content_rating.head())

# Data preprocessing II: audience_status feature
print(f'Audience status category: {df_movie.audience_status.unique()}')

# Visualize the distribution of each category
ax = df_movie.audience_status.value_counts().plot(kind='bar', figsize=(12,9))
ax.bar_label(ax.containers[0])

# Encode audience status feature with ordinal encoding
audience_status = pd.DataFrame(df_movie.audience_status.replace(['Spilled','Upright'],[0,1]))
audience_status.head()

# Data preprocessing III: tomatometer_status feature
# Encode tomatometer status feature with ordinal encoding
tomatometer_status = pd.DataFrame(df_movie.tomatometer_status.replace(['Rotten','Fresh','Certified-Fresh'],[0,1,2]))
tomatometer_status

# Combine all of the features together into one dataframe
df_feature = pd.concat([df_movie[['runtime', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 'audience_count', 'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count']]
                        , content_rating, audience_status, tomatometer_status], axis=1).dropna()
df_feature.head()

# Check the distribution of feature dataframe
df_feature.describe()

# Check class distribution of our target variable:tomatometer_status  
ax = df_feature.tomatometer_status.value_counts().plot(kind='bar', figsize=(12,9))
ax.bar_label(ax.containers[0])

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(df_feature.drop(['tomatometer_status'], axis=1), df_feature.tomatometer_status, test_size= 0.2, random_state=42)
print(f'Size of training data is {len(X_train)} and the size of test data is {len(X_test)}')
