"""Contributors: Zoe Hamel, Samuel Mensah"""

def clean_punctuation(dataset):
    """
    a function that removes all punctuation
    """
    # remove punctuation
    dataset['comment_text'] = [re.sub('[^\w\s]|(\n)',' ', i) for i in dataset['comment_text']]
    return dataset



def data_generator(dataset):
    """
   a function to return x and y
   where
   x = list of comments
   y = a one hot vector of the classes
   """
    # get clean data
    dataset = clean_punctuation(dataset)    

    # create list and array
    x = list(dataset['comment_text'])
    y = np.array(dataset.iloc[:,2:])
    
    return[x,y]
