import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass

@dataclass
class stats:
    P_spam:float
    P_ham:float
    P_word_spam:dict
    P_word_ham:dict


def read_dataset(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def count_vectorize_texts(df:pd.DataFrame) -> pd.DataFrame:
    count_vectorizer = CountVectorizer()
    texts = df['text']
    X = count_vectorizer.fit_transform(texts)

    feature_names = count_vectorizer.get_feature_names_out()

    vectorized_df = pd.DataFrame(X.toarray(),columns=feature_names)

    merged_df = pd.concat([vectorized_df,df],axis=1)
    merged_df.drop('text',axis=1,inplace=True)

    return merged_df

def get_stats(df:pd.DataFrame) -> stats:

    target_dist = df['target'].value_counts().to_dict()
    count_spam = target_dist['spam']
    count_ham = target_dist['not spam']

    spam_df = df[df['target'] == 'spam']
    ham_df = df[df['target'] == 'not spam']

    P_spam = count_spam / (count_spam + count_ham)
    P_ham = count_ham / (count_spam + count_ham)

    P_word_spam = {}
    P_word_ham = {}

    spam_df.drop('target',axis=1,inplace=True)
    ham_df.drop('target',axis=1,inplace=True)

    all_spam = spam_df.sum(axis=0)
    all_ham = ham_df.sum(axis=0)

    for word in all_spam.index:
        P_word_spam[word] = all_spam[word] / all_spam.sum()
        P_word_ham[word] = all_ham[word] / all_ham.sum()

    return stats(
        P_spam = P_spam,
        P_ham = P_ham,
        P_word_spam = P_word_spam,
        P_word_ham = P_word_ham
    )


if __name__ == '__main__':
    path = 'Spam_Detection_Dataset.csv'
    df = read_dataset(path)

    counted_df = count_vectorize_texts(df)

    stats = get_stats(counted_df)

    print(stats)
