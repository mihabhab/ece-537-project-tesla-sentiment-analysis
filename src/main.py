import os
import sys
import pandas as pd
from typing import Callable
# import knime.scripting.io as knio

# *** PATHS ***
script_dir=os.path.dirname(__file__)
data_path = os.path.join(script_dir, '../data/processed/tweets_processed.csv')
BoW = os.path.join(script_dir, '../data/raw/sentiment.xlsx')
# ABBREVIATIONS = 
# STEMWORDS = 
destination_path = os.path.join(script_dir, '../data/processed/scored_dataset.csv')

# FUNCTION DEFINITION
def remove_characters(dataframe:pd.DataFrame, column_name:str, characters:list[str])-> pd.DataFrame:
    """
    Remove specified characters from all cells in a specified column of a DataFrame.
    
    Parameters:
    dataframe (DataFrame): Input DataFrame.
    column_name (str): Name of the column from which to remove characters.
    characters (str): String containing characters to be removed.
    
    Returns:
    DataFrame: DataFrame with specified characters removed from the specified column.
    """
    # Define a function to remove characters from a single cell
    def remove_chars(cell:str)->str:
        for char in characters:
            cell = cell.replace(char, '')
        return cell
    
    # Apply the function to the specified column
    dataframe[column_name] = dataframe[column_name].apply(remove_chars)
    return dataframe

def normalize(dataframe: pd.DataFrame, column_name: str, new_column_name: str, lower_bound: float=0.0, upper_bound: float=1.0) -> pd.DataFrame:
    """
    Normalize the values of a specified column between two input boundaries and create a new column with the normalized values.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column to be normalized.
    lower_bound (float): Minimum value of the desired range.
    upper_bound (float): Maximum value of the desired range.
    new_column_name (str): Name of the new column to be created.
    
    Returns:
    pd.DataFrame: DataFrame with the new normalized column added.
    """
    # Ensure the specified column exists in the DataFrame
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Column range
    column_min = dataframe[column_name].min()
    column_max = dataframe[column_name].max()
    
    # Normalizing
    dataframe[new_column_name] = ((dataframe[column_name] - column_min) / (column_max - column_min)) * (upper_bound - lower_bound) + lower_bound
    
    return dataframe


def sentiment_score(word:str, bucket_of_words: pd.DataFrame, positive_col:str, negative_col:str)-> float:
    """Assigns a score based on the 

    Args:
        word (str): _description_

    Returns:
        float: _description_
    """
    if word in bucket_of_words[positive_col].values:  
        return 0.1 
    elif word in  bucket_of_words[negative_col].values: 
        return -0.1
    else:
        return 0.0


def community_score(dataframe: pd.DataFrame, columns: list[str], weights: tuple[float], new_column_name: str) -> pd.DataFrame:
    """
    Calculate a new score column based on the specified columns and weights.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    columns (List[str]): List of column names containing the values to be multiplied.
    weights (Tuple[float]): Tuple of weights corresponding to each column.
    new_column_name (str): Name of the new column to be created.
    
    Returns:
    pd.DataFrame: DataFrame with the new score column added.
    """
    # Ensure the number of columns matches the number of weights
    if len(columns) != len(weights):
        raise ValueError("Number of columns must match the number of weights.")
    
    # Calculate the new score column based on the equation
    dataframe[new_column_name] = sum(dataframe[col] * weight for col, weight in zip(columns, weights))
    
    return dataframe

def two_column_operation(dataframe: pd.DataFrame, column1_name: str, column2_name: str, new_column_name: str, operation: Callable[[float, float], float]) -> pd.DataFrame:
    """
    Combine two numeric columns using a specified operation or function and create a new column with the result.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame.
    column1_name (str): Name of the first numeric column.
    column2_name (str): Name of the second numeric column.
    operation (Callable[[float, float], float]): Operation or function to combine the values of the two columns.
    new_column_name (str): Name of the new column to be created.
    
    Returns:
    pd.DataFrame: DataFrame with the new combined column added.
    """
    # Ensure the specified columns exist in the DataFrame
    if column1_name not in dataframe.columns or column2_name not in dataframe.columns:
        raise ValueError("Specified columns do not exist in the DataFrame.")
    
    # Combine the values of the specified columns using the specified operation
    dataframe[new_column_name] = operation(dataframe[column1_name], dataframe[column2_name])
    
    return dataframe

# *** PANDAS DATAFRAMES ***
dataset = pd.read_csv(data_path)
sentiment = pd.read_excel(BoW)
print(dataset.columns)

# *** DATASET PRE-PROCESSING ***
characters_to_remove = ['@', '#', '!', '?', '&', '%', '$', '~', '-', '_', "'", '"', "'s"]
dataset = remove_characters(dataset, 'tweet', characters_to_remove)

# *** SENTIMENT SCORE (S1) ***
total_scores= [] 
for index, row in dataset.iterrows(): 
    tweet=row['tweet'] 
    # tweet=row['Preprocessed Document'] 
    words=tweet.split() 
    total_score= sum([sentiment_score(word=word,
                                      bucket_of_words=sentiment,
                                      positive_col='positive_word',
                                      negative_col='negative_word') for word in words]) 
    total_scores.append(total_score) 

dataset['Score1'] = total_scores 

# Normalizing Score2 between specified  upper and lower bounds
dataset = normalize(dataframe=dataset, column_name='Score1',
                    new_column_name='Norm Score1',
                    lower_bound=-1.0,
                    upper_bound=1.0)

# *** COMMUNITY SCORE (S2) ***
# Specify the columns, weights, and the new column name
columns_to_multiply = ['nlikes', 'nreplies', 'nretweets']
count_weights = (0.7, 0, 0.3)
new_column_name = 'Score2'

# Calculate the new score column
dataset = community_score(dataframe=dataset,
                          columns=columns_to_multiply,
                          weights=count_weights,
                          new_column_name=new_column_name)

# Normalizing Score2 between specified  upper and lower bounds
dataset = normalize(dataframe=dataset, column_name='Score2',
                    new_column_name='Norm Score2',
                    lower_bound=-1.0,
                    upper_bound=1.0)

# *** OVERALL SCORE ***
operation = lambda x, y: x * y  # Operation between the two columns
dataset= two_column_operation(dataframe=dataset,
                                   column1_name='Norm Score1',
                                   column2_name='Norm Score2',
                                   new_column_name='Overall Score',
                                   operation=operation)

print(dataset['Overall Score'])
dataset.to_csv(destination_path, index=False)

# Output the modified dataframe 
# knio.output_tables[0] = knio.Table.from_pandas(dataset) 