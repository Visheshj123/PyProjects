#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

csv_path = './movies.csv'
movies_df = pd.read_csv(csv_path)
csv_path = './ratings.csv'
ratings_df = pd.read_csv(csv_path)

print(movies_df.columns)
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
#movies_df.head()

#split genres with the char '|'
movies_df['genres'] = movies_df.genres.str.split('|')
#movies_df.head()

#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.columns)

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)





#------------------------------------------------Content-Based Recommender------------------------------------------------

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

#extract movieId and put it ito userInput Dataframe
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
print(inputMovies.head())

#getting movie data of the movies of interest tothe user
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userMovies.head())



#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable.tail())

#Multiply userinput's reviews by the feature matrix within userMovies to assign weights
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
print(userProfile.head())
print(userProfile.sum())

#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head()) #genre table of a bunch of random movies



#get the weighted average by multiplying the weoghted feature matrix with the genretable, summing it, then dividing it by the userprfile.sum
#Multiply the genres by the weights and then take the weighted average
#weoighted average requries you to divide by the sum of the weights
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())
