import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """ 
    Loads data from given filepaths and merges them. 
  
    Parameters: 
    messages_filepath (str): messages filepath
    categories_filepath (str): categories filepath
    
    Returns: 
    df (DataFrame): combined dataframe
  
    """
    
    # reading csv files using pandas
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merging the two dataframes
    df = pd.merge(messages, categories, on="id")
    
    return df

def clean_data(df):
    
    """ 
    Creates new individual category columns by splitting values 
    from categories column and drops duplicate rows.
  
    Parameters: 
    df (DataFrame): dataframe to be cleaned.
  
    Returns: 
    df (DataFrame): cleaned dataframe with new columns.
  
    """
    
    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # selecting the first row of the categories dataframe to extract column names
    row = categories.iloc[0,:]
    
    # creating a list of category column names
    category_colnames = row.apply(lambda x : x[:-2])
    
    # renaming the columns of 'categories' dataframe
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # replacing categories column in df with new category columns.
    df.drop('categories', inplace=True, axis=1)
    
    # concatenating the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # removing duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    """ 
    Saves the dataframe to a database.
    
    Parameters: 
    df (DataFrame): dataframe to be stored.
    database_filename (str) : database filename.
  
    """
    
    # initiating SQLAlchemy Engine
    engine = create_engine('sqlite:///'+database_filename)
    
    # using pandas to save the DataFrame to the database
    df.to_sql(database_filename[:-3], engine, index=False)  

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()