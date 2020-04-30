import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class model():
	self.stop=stopwords.words("english")
	self.punctuation=string.punctuation
	self.stem=PorterStemmer()
	self.TfidfVectorizer=TfidfVectorizer
	self.word_tokenize=word_tokenize
	self.LogisticRegression=LogisticRegression
	self.MultinomialNB=MultinomialNB
	self.clean="@\S+|https?:\S+|http?:\S|"
	self.full_words={"can't":"can not","aren't":"are not","isn't":"is not","that's":"that is","i'm":"i am","it's":"it is"}
	def preprocess(self,text):
	    corp=re.sub(self..clean,"",str(text).lower())
	    words=word_tokenize(corp)
	    out=""
	    for i in words:
	        if i not in stop and punctuation:
	            out+=" "+stem.stem(i)
	    V=self.TfidfVectorizer(ngram_range=(1,2),max_features=10000,stop_words='english')
	    V=joblib.load()
	    transformed=V.transform(out)
	    return transformed
        

	nb=self.MultinomialNB('alpha': 10.0, 'fit_prior': True)

	lr=self.LogisticRegression(random_state=0,n_jobs=-1)

	def output(self,text):
	    out=self.preprocess(text)
	    result=nb.predict(out)+lr.predict(out)
	    if result == [0]:
	    	return 0
	        # print(f"{text} ==> Negative sentiment")

	    else:
	    	return 1
	        # print(f"{text} ==>Positive sentiment")
