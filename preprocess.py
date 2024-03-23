import pandas as pd
import numpy as np
import nltk
import re
import unicodedata
from textblob import TextBlob
nltk.download('stopwords')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english') # All English Stopwords
# Contaction to Expansion > can't TO can not ,you'll TO you will
contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
# Function to remove Stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]# To remove all stopwords
    return text


def remove_acc_data(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


def tokenize(text):
    tokens = re.split('\W+', text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens



def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

Accuracy_df=pd.read_csv('test.csv',lineterminator='\n',encoding='unicode_escape')
print(Accuracy_df.head())

df1 = Accuracy_df[['text','sentiment']]
df1.columns=['text','sentiment']
print(df1.head())

df1['word_count'] = df1['text'].apply(lambda x : len(str(x).split()))
df1 = df1[df1['text'].notnull()]
df1['stop_words'] = df1['text'].apply(lambda x : len([t for t in x.split() if t in stopword]))
df1['#tag'] = df1['text'].apply(lambda x : len([t for t in x.split() if t.startswith('#')]))
df1['@'] = df1['text'].apply(lambda x : len([t for t in x.split() if t.startswith('@')]))
print(df1.head())
df1['text'] = df1['text'].apply(lambda x : x.lower())
df1['emails'] = df1['text'].apply(lambda i : re.findall(r'([A-Za-z0-9+_-]+@[A-Za-z0-9+_-]+\.[A-Za-z0-9+_-]+)', i))
df1['emails_count'] = df1['emails'].apply(lambda i : len(i))
df1['text'] = df1['text'].apply(lambda i : re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+','', i))
df1['text'] = df1['text'].apply(lambda i : re.sub(r'([A-Za-z0-9+_]+@[A-Za-z0-9+_]+\.[A-Za-z0-9+_]+)','', i))
df1['text'] = df1['text'].apply(lambda d : re.sub('[^A-Z a-z 0-9-]+','', d))
df1['text'] = df1['text'].apply(lambda x : remove_acc_data(x))

print(df1.head())

print(df1.head())
df1['text_x_tokenized'] = df1['text'].apply(lambda x: tokenize(x.lower()))
df1['text_x_lemmatized'] = df1['text_x_tokenized'].apply(lambda x: lemmatizing(x))
print(df1.head())
df1.to_csv('cleaned_sentiment.csv')