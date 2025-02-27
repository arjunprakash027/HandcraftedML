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


class MultinominalNB:

    def __init__(self,df):
        self.df = df


    def get_stats(self,df:pd.DataFrame) -> stats:

        target_dist = df['target'].value_counts().to_dict()
        count_spam = target_dist['spam']
        count_ham = target_dist['not spam']

        spam_df = df[df['target'] == 'spam'].copy()
        ham_df = df[df['target'] == 'not spam'].copy()

        P_spam = count_spam / (count_spam + count_ham)
        P_ham = count_ham / (count_spam + count_ham)

        P_word_spam = {}
        P_word_ham = {}

        spam_df.drop('target',axis=1,inplace=True)
        ham_df.drop('target',axis=1,inplace=True)

        #+1 is for laplase smoothning to ensure if the word is not present, the score does not go to 0
        all_spam = spam_df.sum(axis=0) + 1
        all_ham = ham_df.sum(axis=0) + 1

        for word in all_spam.index:
            P_word_spam[word] = all_spam[word] / all_spam.sum()
            P_word_ham[word] = all_ham[word] / all_ham.sum()

        return stats(
            P_spam = P_spam,
            P_ham = P_ham,
            P_word_spam = P_word_spam,
            P_word_ham = P_word_ham
        )
    
    def fit(self) -> None:
        self.stats = self.get_stats(self.df)
    
    def predict(self,text:str) -> dict:
        P_spam = self.stats.P_spam
        P_ham = self.stats.P_ham
        P_word_spam = self.stats.P_word_spam
        P_word_ham = self.stats.P_word_ham

        words = text.split()

        spam_pred = np.log(P_spam)
        ham_pred = np.log(P_ham)

        for word in words:
            observed_prob_spam = P_word_spam.get(word,1)
            observed_prob_ham = P_word_ham.get(word,1)

            spam_pred += np.log(observed_prob_spam)
            ham_pred += np.log(observed_prob_ham)
        
        return {
            "spam":spam_pred,
            "ham":ham_pred
        }

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


if __name__ == '__main__':
    path = 'Spam_Detection_Dataset.csv'
    df = read_dataset(path)

    counted_df = count_vectorize_texts(df)

    model = MultinominalNB(counted_df)
    model.fit()

    print(model.predict("more what"))

