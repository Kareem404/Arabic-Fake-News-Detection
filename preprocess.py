import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import torch
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer

# run this after the installing the requirements file: 
# pip install torch==2.5.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

def read_dataset(dataset_dir:str='.\AFND\Dataset') -> np.array:
    "Function that reads the dataset and returns a numpy array"

    # read sources.json
    sources_file_path = '.\AFND\sources.json'
    with open(sources_file_path, 'r', encoding='utf-8') as sources_file:
        sources_data = json.load(sources_file)
    sources_df = pd.DataFrame(list(sources_data.items()), columns=['source', 'label'])

    # Read scraped_articles.json for each source
    articles_data = []

    for source in sources_df['source']:
        scraped_articles_path = os.path.join(dataset_dir, source, 'scraped_articles.json')
        
        # Check if the file exists before attempting to read it
        if os.path.exists(scraped_articles_path):
            with open(scraped_articles_path, 'r', encoding='utf-8') as articles_file:
                source_articles_dict = json.load(articles_file)
                source_articles_list = source_articles_dict.get("articles", [])
                
                # Add a 'source' key to each article
                for article in source_articles_list:
                    article['source'] = source
                
                articles_data.extend(source_articles_list)

    # Convert articles_data to a DataFrame
    articles_df = pd.DataFrame(articles_data)

    merged_df = pd.merge(articles_df, sources_df, how='inner', left_on='source', right_on='source')

    # drop the published_date and source featrues
    news = merged_df.drop(columns=['published date', 'source'])

    # remove the "undecided" sources
    binary_news = news[news['label'] != 'undecided']

    print(f'{len(binary_news[binary_news['label'] == 'credible'])} credible articles have been read successfully')

    print(f'{len(binary_news[binary_news['label'] == 'not credible'])} not credible articles have been read successfully')
    
    return np.array(binary_news)


def split_and_preprocess(dataset:np.array, max_len:int = 350) -> tuple[list]:
    """Function that preprocesses and splits the dataset
    args:-
    dataset: numpy array representing the dataset
    max_len: the maximum length of an article text in the dataset
    returns:-
    tuple representing the the splitted text and title of articles"""

    titles = dataset[:, 0]
    texts = dataset[:, 1]
    target = dataset[:, 2]

    df_titles = pd.DataFrame(titles, columns=['Title'])
    df_text = pd.DataFrame(texts, columns=['Text'])
    df_target = pd.DataFrame(target, columns=['Target'])

    dataset = pd.concat([df_titles, df_text, df_target], axis=1)

    # remove rows where the length of df_text exceeds 350
    dataset = dataset[dataset['Text'].apply(lambda x: len(x.split()) <= max_len)]

    X = dataset[['Title', 'Text']]
    y = dataset['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random_state = 42

    X_train_title = list(X_train['Title'])
    X_test_title = list(X_test['Title'])

    X_train_text = list(X_train['Text'])
    X_test_text = list(X_test['Text'])

    ### Preprocess the dataset
    model_name="aubmindlab/bert-base-arabertv02"
    arabert_prep = ArabertPreprocessor(model_name=model_name)

    X_train_title = list(map(lambda x: arabert_prep.preprocess(x), X_train_title))
    X_test_title = list(map(lambda x: arabert_prep.preprocess(x), X_test_title))
    X_train_text = list(map(lambda x: arabert_prep.preprocess(x), X_train_text))
    X_test_text = list(map(lambda x: arabert_prep.preprocess(x), X_test_text))

    return X_train_title, X_test_title, X_train_text, X_test_text, y_train, y_test

def tokenize_text(texts, max_size):
    arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    return arabert_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=max_size,
    ).to('cuda') 


# main program 
if __name__ == '__main__':

    # read the dataset
    binary_news = read_dataset(dataset_dir='.\AFND\Dataset')
    X_train_title, X_test_title, X_train_text, X_test_text, y_train, y_test = split_and_preprocess(dataset=binary_news, 
                                                                                                    max_len=350) # 350
    # tokenizer the text
    X_train_title_tokenize = tokenize_text(X_train_title, max_size=20)
    X_train_text_tokenize = tokenize_text(X_train_text, max_size=280) # 280
    X_test_title_tokenize = tokenize_text(X_test_title, max_size=20)
    X_test_text_tokenize = tokenize_text(X_test_text, max_size=280) # 280

    # create a directory
    try:
        os.mkdir('Split')
        print(f'Created directory successfully')
    except FileExistsError:
        pass
    
    # save the tensors in the created directory
    torch.save(X_train_title_tokenize, "./Split/X_train_title_tokenize.pt")
    torch.save(X_test_title_tokenize, "./Split/X_test_title_tokenize.pt")
    torch.save(X_train_text_tokenize, "./Split/X_train_text_tokenize.pt")
    torch.save(X_test_text_tokenize, "./Split/X_test_text_tokenize.pt")
    torch.save(y_train, "./Split/y_train.pt")
    torch.save(y_test, "./Split/y_test.pt")
    
    print('Saved the training and testing data successfully!')