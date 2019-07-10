import sys
import nltk
nltk.download(['stopwords','punkt','wordnet','averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import pickle
from FeatureExtractor import StartingVerbExtractor

def load_data(database_filepath):
    '''
     input: (
            database_filepath: file path to load data from 
            )
     Function loads data from the database and saves them in a dataframe and splits the data into the message dataframe and the categories dataframe 
     output: (
        X: Dataframe containing all messages
        Y: Dataframe categories
        categories: list all categories names
        )
    '''

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    messages_select_query = "Select * from messages" 
    df = pd.read_sql("select * from messages",con=engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'],axis=0)
    return X, Y, Y.columns

def tokenize(text):
    '''
    input: (
        text: string to tokenize 
        )
    Function reads a string, cleans it and returns a list of extracted tokens after removing stop words and short words and replacing all urls with a urlplaceholder. 
    Then generalizes all word forms by lemmetization  
    output: (
        clean_tokens: list of tokens in text after cleaning
        )
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    stop_words = stopwords.words("english")
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words: 
        if word not in stop_words:
            clean_tok = lemmatizer.lemmatize(word).lower().strip() 
            clean_tokens.append(clean_tok)
    # remove short words
    clean_tokens = [token for token in clean_tokens if len(token) > 2]
    return clean_tokens


def build_model():
    '''
    Function builds a model pipeline using random forest classifier and runs GridSearch to optimze classifier parameters
    output: (
        cv: model pipeline after optimization
        )
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor(tokenize=tokenize))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # run grid_search to find best value for n_estimators and min_samples_split 
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__min_samples_split': [2, 4, 6]
    }

    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=4, verbose=2)

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input:(
        model: pipeline trained classifier model,
        X_test: inpute features for testing the model,
        Y_test: output categories for evaluating the model,
        category_names: list of output categories or labels
    )
    Function gets the trained model and run predictions on it using the testing dataset then evaluates the model prints the classification report
    
    '''
    y_pred = model.predict(X_test)
    for ndx in range(0,len(category_names)):
        print(category_names[ndx])
        print(classification_report(Y_test.iloc[:,ndx],y_pred[:,ndx]))

def save_model(model, model_filepath):
    '''
    input:(
        model: pipeline trained classifier model,
        model_filepath: filepath pickle file to save the trained model
    )
    Function saves the trained model in a pickle file for future predictions
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
