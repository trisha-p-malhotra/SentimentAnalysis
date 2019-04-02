import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("./training.1600000.processed.noemoticon.csv", header=None, names=cols, encoding='ISO-8859-1')

df.drop(['id','date','query_string','user'],axis=1,inplace=True)

#print(df.head())
#print(df.sentiment.value_counts())

df[df.sentiment == 0].head(10)
df[df.sentiment == 4].head(10)



from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))



df['pre_clean_len'] = [len(t) for t in df.text]

# data dictionary
from pprint import pprint
data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}

pprint(data_dict)
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

import re

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
print(test_result)

print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(0, 1600000):
    #if((i+1)%10000 == 0):
    #    print("Tweets %d of %d has been processed")
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

print(str(len(clean_tweet_texts)) + "tweets cleaned")

clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()
clean_df.to_csv('clean_tweet.csv',encoding='utf-8')

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
print(my_df.head())