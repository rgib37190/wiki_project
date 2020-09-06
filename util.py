import re
from nltk.corpus import stopwords

def clean_not_english_word(x):
    if x is not None:
        x = re.sub("[^a-zA-Z.']+",' ',x).strip()
        return x 
    else:
        return None
    
def clean_stopwords(x):
    stopwords_list = stopwords.words('english')
    if x is not None:
        word_list = []
        for word in x.split(' '):
            if word not in stopwords_list:
                word_list.append(word)
        new_sentence = ' '.join(word_list)
        return new_sentence
    else:
        return None
    
def clean_special_word(x):
    if x is not None:
        x = re.sub("[（）]+",' ',x).strip()
        return x 
    else:
        return None
    
def clean_comma(x):
    if x is not None:
        x = re.sub("[，。]+",' ',x).strip()
        return x 
    else:
        return None
    
