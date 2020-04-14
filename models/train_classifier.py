# import libraries
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):

    """ 
    Loads data from database. Also splits data into 
    predictor and target variables and returns them.
  
    Parameters: 
    database_filepath (str): Database filepath
    
    Returns: 
    X (DataFrame): Feature Columns
    Y (DataFrame): Label Columns
    category_names (List): Category Names List
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)

    # using pandas to read table from database
    df = pd.read_sql_table(database_filepath[:-3], engine)

    # splitting X and Y
    X = df.message.values
    Y = df.iloc[:,4:].values

    return X, Y, df.columns[4:].values


def tokenize(text):

    """ 
    Cleans data using various NLP techniques. 
  
    Parameters: 
    text (str): Text For Cleaning And Tokenizing (English).
    
    Returns: 
    text (List): Tokenized Text, Clean For ML Modeling
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = [t for t in word_tokenize(text) if t not in stopwords.words('english')]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


def build_model():

    """
    Creates pipeline and builds the model.

    Returns: 
    model (GridSearchCV or Scikit Pipeline Object) : ML Model
    """

    # creating multioutput classifier pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters to grid search
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split':[2,3]
    }
    
    # initiating GridSearchCV method
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    
    """ 
    Evaluates the model and predicts the category.
  
    Parameters:
    model (GridSearchCV or Scikit Pipeline Object) : Trained ML Model
    X_test (DataFrame) : Test Features
    Y_test (DataFrame) : Test Labels
    category_names (List): Category Names List
    """

    # predict on test data
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test[i], y_pred[i]))
        print('Accuracy of %2s: %.2f' %(category_names[i], accuracy_score(y_test[i], y_pred[i])))
        print('--------------------------------------------------------')


def save_model(model, model_filepath):

    """
    Pickles the model and stores in the specified path.
    
    Parameters:
    model (GridSearchCV or Scikit Pipeline Object) : Trained ML Model
    model_filepath (str) : Destination Path To Save .pkl File
    """
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]
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