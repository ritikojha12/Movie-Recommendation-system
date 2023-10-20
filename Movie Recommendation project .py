#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity
import os


# In[2]:


display (os.getcwd())


# In[3]:



movies_data =pd.read_csv("C:\\Users\\ritik\\Downloads\\movies.csv")
movies_data.head()



# In[4]:


display (movies_data.shape)


# In[5]:


selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[6]:


display (movies_data.info())


# In[7]:


display (movies_data.isna().sum())


# In[8]:


display (movies_data[selected_features].head())


# In[9]:


display (movies_data[selected_features].isna().sum())


# In[10]:



for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
display (movies_data.head())


# In[11]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
display (combined_features)


# In[12]:


# Vector shape is (4803, 17318). This is based on the number of distinct words. All the words will be converted to their equivalent numbers.

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
display (feature_vectors.shape)
print (feature_vectors)


# In[13]:


similarity = cosine_similarity(feature_vectors)
print  (similarity )


# In[14]:


# display


# In[15]:


display(similarity.shape)


# In[17]:


pd.DataFrame(similarity).to_csv("C:\\Users\\ritik\\Downloads\\12_may_Movie.csv")


# In[28]:


movie_name = input(' Enter your favourite movie name : ')


# In[29]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[31]:


len(list_of_all_titles)


# In[30]:



find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[37]:


close_match = find_close_match[0]
print(close_match)


# In[38]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[33]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)



# In[25]:


len(similarity_score)


# In[34]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[36]:


print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




