"""Contributors: Zoe Hamel, Samuel Mensah"""

def data_generator(dataset):
    """
   a function to return x and y
   where
   x = list of comments
   y = a one hot vector of the classes
   """

    # remove punctuation
    dataset['comment_text'] = [re.sub('[^\w\s]|(\n)',' ', i) for i in dataset['comment_text']]

    # create list and array
    x = list(dataset['comment_text'])
    y = np.array(dataset.iloc[:,2:])
    
    return[x,y]
