from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import pandas as pd

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenize):
        self.tokenize = tokenize
        
    def starting_verb(self, text):
        '''
        input: (
            self: class object instance,
            text: string text to extract the starting verb 
            )
        Method reads text data and returns True if the text starts with a verb and False otherwise  
        output: (
        found: Boolean True if data starts with a verb and False otherwise
        )
        '''
        sentence_list = nltk.sent_tokenize(text)
        found = False
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenize(sentence))
            
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    found = True
                    break
        return found

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)