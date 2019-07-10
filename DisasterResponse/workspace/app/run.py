import json
import plotly
import pandas as pd
import nltk
nltk.download('stopwords','punkt','wordnet','averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine 
import sys
import os
sys.path.append(os.path.abspath('../models'))
from FeatureExtractor import StartingVerbExtractor

app = Flask(__name__)

def takeSecond(elem):
    '''
    input:(
        elem: set of two elemets,
    )
    Function used by the sorted() function to get the second elemnt in a set
    output:(
        elem[1]: the second element of the set
    )
    '''
    return elem[1]

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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [token for token in clean_tokens if len(token) > 2]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Function handels the '/index' route and renders web page with plotly graphs after creating graphJSON object 
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # get message categories
    categories = df.drop(columns=['id','message','original','genre'])
    categories_count = [(col,categories[col].sum() ) for col in categories.columns]
    
    # get top categories correlation
    top_categories = sorted(categories_count,key=takeSecond,reverse=True)[:10]
    top_categories_names = [ cat[0] for cat in top_categories]
    top_categories_corr = df[top_categories_names].corr()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[item[0] for item in categories_count],
                    y=[item[1] for item in categories_count]
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=top_categories_corr.columns,
                    y=top_categories_corr.columns,
                    z=top_categories_corr.values
                )
            ],

            'layout': {
                'title': 'Top Categories Corrolation Heatmap',
                'yaxis': {
                    'title': "Top Categories"
                },
                'xaxis': {
                    'title': "Top Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    Function handels the '/go' route, get the input queru from request, predict the classification tables and renders 
    web page that displays the predicted clssification labels
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()