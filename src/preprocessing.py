"""Contributors: Samuel Mensah"""

def data_generator(dataset):
    """
    a function to return x and y
    where
    x = list of comments
    y = a one hot vector of the classes
    """
    x = np.array(df['comment_text'])
    x.tolist()
    y = np.array(dataset.iloc[:,2:])
    return(x,y)

