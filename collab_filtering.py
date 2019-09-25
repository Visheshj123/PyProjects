
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




#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

print(movies_df.head())

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

#------Collaborative Filtering with Pearson Correlation--------------------------------
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

#get movie ID's of input users movies

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)



#get users who have the same movies
#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
print(userSubsetGroup.head())

userSubsetGroup.get_group(1130)

#Sorting it so users with movie most in common with the input will have priority
#this is done by getting the group with the largest number of movies in common and putting it at the top
#this is what len(x[1]) since column 1 is the movieID column
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
print(userSubsetGroup[0:3])


#obtains first 10 groups
userSubsetGroup = userSubsetGroup[0:100]


#Store the Pearson Correlation in a dictionary, where the key is the outisder user Id and the value is the coefficient
pearsonCorrelationDict = {}




#Iterates through one group at a time where name is the name of the group and group is the data
for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)

     #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    tempRatingList = temp_df['rating'].tolist()

     #Let's also put the current outside user's reviews in a list format
    tempGroupList = group['rating'].tolist()

    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)


    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0


#print(pearsonCorrelationDict.items())


#make pearson correaltion into a dataframe
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

#top 50 users
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

#new DF that has the outside user's movie ratings next othe pearson correlation
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())


#to get the weighted average, we need to multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()


#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)



movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

print(movies_df[0:10])
