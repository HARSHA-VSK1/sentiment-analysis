import numpy as np
import pandas as pd

# data processing/manipulation
pd.options.mode.chained_assignment = None
import re

# data visualization
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import plotly.express as px

# stopwords, tokenizer, stemmer
import nltk
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist


# spell correction, lemmatization
from textblob import TextBlob
from textblob import Word

# sklearn
from sklearn.model_selection import train_test_split

classifier = pipeline("sentiment-analysis", framework="pt", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)
#classifier1=pipeline('fill-mask', model='roberta-base')

#print(Accuracy_df)
def Sentiment_Analysis1(df, batch_size=1):            #function to run sentiment analysis
    Tweet_list = df['text'].tolist()
    results = []
    for i in range(0, len(Tweet_list), batch_size):
        batch = Tweet_list[i:i+batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)
    labels = [result['label'].replace('LABEL_0', 'negative').replace('LABEL_1', 'neutral').replace('LABEL_2', 'positive') for result in results]
    scores = [result['score'] for result in results]
    print(scores)
    df['label'] = labels
    df['score'] = scores
    return df

df=pd.read_csv('cleaned_sentiment.csv',lineterminator='\n',encoding='unicode_escape')
newdf1 = Sentiment_Analysis1(df.iloc[0:3000,:])
print(newdf1.head())
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
accuracy=accuracy_score(newdf1['sentiment'],newdf1['label'])
print(accuracy)
cm=confusion_matrix(newdf1['sentiment'],newdf1['label'])
sns.heatmap(cm,
			annot=True,
			fmt='g')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()
