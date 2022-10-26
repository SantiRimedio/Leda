import requests
import tweepy
import pandas as pd
import openpyxl

client = tweepy.Client( bearer_token="AAAAAAAAAAAAAAAAAAAAAGvYiAEAAAAA%2BhyGSMkt2KMdKWabOVNr5eiV3zo%3DS4EAPg7zquxLLq2qjwxWnNLuYCr3Gz6obG7wEOKn2la76dST2M",
                        consumer_key="VSEZwoCdsaawnTCrxmG1dJTIe",
                        consumer_secret="JmlWELxvKTWDHou2VeB3ECp97t5c5SWnAKh7TqWkuB53cCP6Gm",
                        access_token="1581351580468711426-2oEGAte7CJFrNK1s2KeaPeFuKtlWBy",
                        access_token_secret="8x4NBnrqjROUB6tLD3MHJA2dqZbbRVUfERsHSykuwWzqut",
                        return_type = requests.Response,
                        wait_on_rate_limit=True)

# Define query
query = 'gran hermano'

tweets = client.search_recent_tweets(query=query,
                                     max_results=100)

#tweets = client.get_recent_tweets_count(query=query, granularity="day")



# Save data as dictionary
tweets_dict = tweets.json()

# Extract "data" value from dictionary
tweets_data = tweets_dict['data']

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data)

df['text'] = df['text'].astype(str).str.lower()
df.head(3)

#Importar ntlk
from nltk.tokenize import RegexpTokenizer
regexp = RegexpTokenizer('\w+')

#Tokenización
df['text_token']=df['text'].apply(regexp.tokenize)
print(df.head(3))

import nltk
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english', 'spanish')
from nltk.corpus import stopwords

# Make a list of spanish stopwords
stopwords = nltk.corpus.stopwords.words('spanish')

# Extend the list with your own custom stopwords
my_stopwords = ['https', "rt"]
stopwords.extend(my_stopwords)

# Remove stopwords
df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
print(df.head(3))

# REmove infrequent words
df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
df[['text', 'text_token', 'text_string']].head()

# All worlds list
all_words = ' '.join([word for word in df['text_string']])

# Tokenize all worlds
nltk.download('punkt')
tokenized_words = nltk.tokenize.word_tokenize(all_words)


#Frequency distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_words)
print(fdist)

#exclude rare words
df['text_string_fdist'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))

#lematización
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
wordnet_lem = WordNetLemmatizer()
df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)

import wordcloud
import matplotlib

all_words_lem = ' '.join([word for word in df['text_string_lem']])

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=600,
                     height=400,
                     random_state=2,
                     max_font_size=100).generate(all_words_lem)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');
plt.show()

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['polarity'] = df['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))
df.tail(3)


df = pd.concat(
    [df.drop(['id', 'polarity'], axis=1),
     df['polarity'].apply(pd.Series)], axis=1)
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(df.loc[df['compound'].idxmax()].values)



#df.to_csv("tweets.csv")
df.to_csv("tweets.csv")

import seaborn as sns

sns.countplot(y='sentiment',
             data=df,
             palette=['#b2d8d8',"#008080", '#db3d13']
             )
plt.show
