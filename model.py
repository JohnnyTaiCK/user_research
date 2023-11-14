from transformers import pipeline
import nltk
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords
import gzip
import json
import pandas as pd
import re
from transformers import AutoTokenizer
import string

#nltk.download()
# nltk.download("stopwords")
# nltk.download('punkt')
# nltk.download('wordnet')

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# text-classification
# sentiment-analysis

def getClassifier(form: str,model_name: str):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return pipeline(form, model=model_name, tokenizer=tokenizer)

def parseDataSet(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parseDataSet(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def text_processing(text: str):
  text = text.lower()
  text = "".join([char for char in text if char not in string.punctuation])
  words = nltk.word_tokenize(text)
  stop_words = stopwords.words('english')
  filtered_words = [word for word in words if word not in stop_words]
  porter = PorterStemmer()
  stemmed = [porter.stem(word) for word in filtered_words]
  return " ".join(stemmed)#List

def preprocess_pipe(texts):
  preproc_pipe = []
  texts = texts.to_list()
  
  for doc in texts:
    preproc_pipe.append(text_processing(str(doc)))
  return preproc_pipe

def defaultModel():
  classifier = getClassifier("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
  df = getDF('AMAZON_FASHION_5.json.gz')
  df.drop(labels=['image'], axis=1, inplace=True)
  threshold = 3
  df["rating_class"] = df["overall"].apply(lambda x: "good" if x > 3 else "bad") #1 = good, 0 = bad

  df = df.rename(columns = {"overall":"rating"})
  df["reviewTime"] = pd.to_datetime(df["reviewTime"])

  df['rev_year'] = df['reviewTime'].dt.year
  df['rev_month'] = df['reviewTime'].dt.month
  Yearly_avg_rating = df.groupby("rev_year")["rating"].mean().reset_index()
  Yearly_avg_rating = Yearly_avg_rating.rename(columns = {"rating":"avg_rating"})
  

  df["label"] = df["reviewText"].apply(lambda comment: classifier(str(comment))[0]["label"])

  df["clean_summary"] = preprocess_pipe(df["reviewText"])

  feq = pd.Series(" ".join(df["clean_summary"]).split()) #like a list of words  
  tagged_words = pos_tag(feq)
  critical_words = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('J')]
  critical_words = pd.Series(critical_words)
  return df

def LiYuanModel():
  classifier = getClassifier("text-classification", "LiYuan/amazon-review-sentiment-analysis")
  df = getDF('AMAZON_FASHION_5.json.gz')
  df.drop(labels=['image'], axis=1, inplace=True)

  df["label"] = df["reviewText"].apply(lambda comment: classifier(str(comment))[0]["label"])
  df['label'] = df['label'].str.extract(r'(\d+)').astype(int)
  return df

LiYuanModel()
# df.to_csv("cleaned_data.csv", sep=',', encoding='utf-8', index = False)


# grouped_df = df.groupby("rating_class")

# good_rev = grouped_df.get_group("good")["clean_summary"]
# good_rev = good_rev.astype("str")
# feq_good_rev = pd.Series(" ".join(good_rev).split()).value_counts()
# print(feq_good_rev[0:12])

# print("\n\n")

# bad_rev = grouped_df.get_group("bad")["clean_summary"]
# bad_rev = bad_rev.astype("str")
# feq_bad_rev = pd.Series(" ".join(bad_rev).split()).value_counts()
# print(feq_bad_rev[0:12])


# reply = client.chat.completions.create(
#             # engine=self.model,
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "user", "content": """please give some recommendations to the Amazon sellers based on the 
#                  following:
#                  1. fast shipment
#                  2. quality products
#                  """},
#             ],
#         )

# print(reply)
