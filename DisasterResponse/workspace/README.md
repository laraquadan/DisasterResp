# Disaster Response Pipeline Project:
This project aims on analyzing real messages that were sent during disaster events. It consists of three parts: 
1. ETL Pipeline: which loads the disaster response messages and thier categories, cleans the data and saves it sql database.
2. ML Pieline the loads the data from the database and builds a model pipleline using the random forest classifier, trains the model on the data and saves the model to a pickle file. 
3. Web application: where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Project Files:
1. ETL Pipeline files in `workspace/data`:
	- `DisasterResponse.db`: database file where clean data is stroed
	- `disaster_messages.csv`: csv file storing all the messages
	- `disaster_categories.csv`: csv file stroing categories of messages
	- `process_data.py`: python script that loads the data, cleans it and saves it to the database

2. ML Pipeline files in `workspace/model`:
	- `FeatureExtractor.py`: python class that creates a custom startng verb feature extractor.
	- `train_classifier.py`: python script that loads the data from the database, creates the model pipeline, trains the data using the model and saves the trained model to c pickle file "classifier.pkl".

3. Web application in `workspace/app`:
	- `run.py`: a Flask script to handel the web application routes and provide the routes with necessary data.
	- `templates/master.html`: the main page in the web application that display plotify figures for data visulization. It also has a search box for workers to enter messages for predictions.
	- `template/go.html`: displays the predicted catgrories for the message entered by the worker.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web appcd.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Acknowledgment
This project is part of [Udacity Datascientist Nanodegree program.](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
Thanks to Udacity for thier great work.
The dataset it provided by [Figure Eight](https://www.figure-eight.com/)