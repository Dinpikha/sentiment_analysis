import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
df=pd.read_csv("test.csv",encoding='latin-1')
print(df.columns)

stopwords=set(stopwords.words('english'))

def removestopwords(text):
    words=word_tokenize(text)
    filteredwords=[word for word in words if word.lower() not in stopwords and word.isalpha()]
    return ' '.join(filteredwords)


if 'text' in df.columns: 
    df['text']=df['text'].fillna('').astype(str)

df['texts']=df['text'].apply(removestopwords)
print(df['texts'])
colums_to_drop=['textID','Time of Tweet', 'Age of User',
       'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)','text']
df=df.drop(colums_to_drop,axis=1)
print(df.columns)
df.to_csv('final.csv',index=False)