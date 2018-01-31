"""Contributors: Zoe Hamel, Samuel Mensah, Buri Gershom
"""
import pandas as pd
import re
import numpy as np

def clean_punctuation(csv_file):
    """
   a helper function that removes all punctuation
   this function takes in dataset as argument.
   where dataset is a csv file
   """
    
    dataset = pd.read_csv(csv_file)
   
    # remove punctuation
    # a list comprehension to remove punctuations
    dataset['comment_text'] = [re.sub('[^\w\s\']|(\n)',' ', i) for i in dataset['comment_text']]
    dataset['comment_text'] = [re.sub( '\s+', ' ', i).strip() for i in dataset['comment_text']]
    
    return dataset

def data_generator(csv_file):
    """
   a function to return x and y
   this function takes in dataset as argument.
   
   Input:
   csv_file: csv file containing the dataset
   
   Output:
   A list containing:
   x = list of comments, 
   y = a one hot vector of the classes
   """
    # get clean data: applying the function clean_punctuation
    dataset = clean_punctuation(csv_file)
    dataset['Clean'] = np.where((dataset['toxic']==0) & (dataset['insult']==0) & (dataset['identity_hate']==0) 
             & (dataset['obscene']==0) & (dataset['severe_toxic']==0) & (dataset['threat']==0), 1, 0)    

    # create list and array
    x = list(dataset['comment_text'])
    y = np.array(dataset.iloc[:,2:])
    return [x,y]
