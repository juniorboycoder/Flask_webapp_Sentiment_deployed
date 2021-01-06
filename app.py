
from flask import Flask, render_template
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, EntitiesOptions, KeywordsOptions, SentimentOptions,CategoriesOptions,EmotionOptions
from flask import redirect,request
import pickle

import nltk
nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')

from io import BytesIO

from nltk.sentiment.vader import SentimentIntensityAnalyzer


from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import pandas as pd
import numpy as np

import boto3
import os
from os import environ

from os import getenv


app= Flask(__name__)

#define keys as enviroment variable
# use set commad on windows cmd

aws_access_key_id = getenv('aws_access_key_id', None)
aws_secret_access_key = getenv('aws_secret_access_key', None)




#app.config['aws_access_key_id']= environ.get('aws_access_key_id')

#app.config['aws_secret_access_key']= environ.get('aws_secret_access_key')

a= aws_access_key_id 
b=aws_secret_access_key



def func(x):
    if x > 0:
        return "positive"
    elif x == 0:
        return "Neutral"
    else:
        return 'Negative'
#templates rendering

@app.route('/',methods=['GET','POST'])

def index():
    
    message = ''
    message2= ''
    message3= ''
    if request.method == 'POST':
      if request.form.get("submit_a"):
        #get text from the form
        text = request.form.get('text')  # access the data inside 
        #to get words : if the text is less tha 10 letters
        if text == request.form.get('text')[:10]:
            #use the vadet sentiment analyzer
            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text)

            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            if text == text:
                message=text
                message3=g[0]
        # using the IBM Natural language understanding API

        elif text == request.form.get('text'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator(' your IAMAuthenticator key ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )
        #API endpoint
            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
        #Response   
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=1),
                keywords=KeywordsOptions(emotion=True, sentiment=True,
                                        limit=2),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions())).get_result()

            print(json.dumps(response, indent=2))
           
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
            

            if text == text:
                message = text
                message2=result2
                
      elif request.form.get("submit_b"):
        # access the data inside 
        text2 = request.form.get('text2')
       
        if text2 == request.form.get('text2')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text2)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if text2 == text2:
                 message=text2
                 message3=g[0]
        elif text2 == request.form.get('text2'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text2,
                features=Features(
                entities=EntitiesOptions( sentiment=True),
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions())).get_result()

            print(json.dumps(response, indent=1))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            


            if text2 == text2:
                message = text2
                message2=result2
                
                
      elif request.form.get("submit_c"):
        # access the data inside 
        text3 = request.form.get('text3')
       
        if text3 == request.form.get('text3')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text3)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if text3 == text3:
                 message=text3
                 message3=g[0]
        elif text3 == request.form.get('text3'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text3,
                features=Features(
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions(),
                emotion=EmotionOptions())).get_result()

            print(json.dumps(response, indent=2))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
            

            if text3 == text3:
                message = text3
                message2=result2
                  
      elif request.form.get("submit_d"):
        # access the data inside 
        text4 = request.form.get('text4')
       
        if text4 == request.form.get('text4')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text4)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if text4 == text4:
                message=text4
                message3=g[0]
        elif text4 == request.form.get('text4'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text4,
                features=Features(
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions(),
                
                categories=CategoriesOptions())).get_result()

            print(json.dumps(response, indent=2))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()

            

            if text4 == text4:
                message = text4
                message2=result2

      elif request.form.get("submit_e"):        
        text5 = request.form.get('text5')
       
        if text5 == request.form.get('text5')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text5)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if text5 == text5:
                 message=text5
                 message3=g[0]
        elif text5 == request.form.get('text5'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text5,
                features=Features(
               
                sentiment=SentimentOptions(),
                keywords=KeywordsOptions(),
                categories=CategoriesOptions())).get_result()

            print(json.dumps(response, indent=2))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()

           
            #GET TEXT AND SUMMARIZE USING GENSIM
            resposesum=summarize(text5, ratio=0.5)

          
                
            #TO PRINT OUT ON TEMPLATE
            if text5 == text5:
                message = text5
                message2=result2
                message3=json.dumps(resposesum, indent=5)
               
     
      elif request.form.get("submit_f"):        
        text6 = request.form.get('text6')
      
        if text6 == request.form.get('text6'):
           
            

         #using aws s3 for saved pickle file

        #use boto3 to load in pickle file from aws S3 bucket 
          #  s3 = boto3.resource('s3',aws_access_key_id=a, aws_secret_access_key= b
         ##  with open('hotelll.pickle', 'wb') as data:
          #      s3.Bucket("projectsss").download_fileobj("hotelll.pickle", data)

         #   with open('hotelll.pickle', 'rb') as data:
                resposesum = pickle.load(data)

         
            #predict sentiment of text
         #   resposesum=resposesum.predict([text6])     

        #OR training online
            #stem words
            stemmer = SnowballStemmer('english')

            #stop words removal
            words = stopwords.words("english")

            df=pd.read_csv('https://projectsss.s3.us-east-2.amazonaws.com/hotel-reviewss.csv')

            #regular expression to clean dataset remove space and convert to uppercase
            df['cleaned'] =df['Description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

            #split data
            X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['Is_Response'], test_size=0.2)

            #NLP Trainer
            pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                ('chi',  SelectKBest(chi2, k=10000)),
                                ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

            model = pipeline.fit(X_train, y_train)
            resposesum=model.predict([text6])   

            if text6 == text6:
                message = text6
               
                message3=resposesum[0]
     



    return render_template('base.html', message=[message] ,message2= [message3,message2])




#function for the api


@app.route('/Word_Sentiment/<text>' ,methods=['GET'])
def Word_Sentiment(text):
    if request.method == 'GET':
        if text == text[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": text,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if text == text:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('Your IAMAuthenticator key')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(
                entities=EntitiesOptions( sentiment=True),
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions())).get_result()

            print(json.dumps(response, indent=1))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
           

            xx = {
            "word": text,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if text == text:
           
                message=yy
               
            return message 
    
    return message
   

@app.route('/Word_Sentiment_and_Emotions/<text>' ,methods=['GET'])
def Word_Sentiment_and_Emotions(text):
    if request.method == 'GET':
        if text == text[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": text,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if text == text:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions(),
                emotion=EmotionOptions())).get_result()

            print(json.dumps(response, indent=1))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
           
            xx = {
            "word": text,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if text == text:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Word_Sentiment_and_Category/<text>' ,methods=['GET'])
def Word_Sentiment_and_Category(text):
    if request.method == 'GET':
        if text == text[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": text,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if text == text:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(
                keywords=KeywordsOptions( sentiment=True),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions())).get_result()
            print(json.dumps(response, indent=1))
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
           
            xx = {
            "word": text,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if text == text:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Word_Sentiment_Emotions_and_Category/<text>' ,methods=['GET'])
def Word_Sentiment_Emotions_and_Category(text):
    if request.method == 'GET':
        if text == text[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(text)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": text,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if text == text:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('your IAMAuthenticator ')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=1),
                keywords=KeywordsOptions(emotion=True, sentiment=True,
                                        limit=2),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions())).get_result()
            result=json.dumps(response, indent=1)
            result1=result.replace('"language": "en"', '')
            result11=result1.replace('"count": 1', '')
            result111=result11.replace('document', 'The_Whole_Sentence')
            result2=result111[80:].splitlines()
            
           
            xx = {
            "word": text,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if text == text:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Hotel_Sentiment/<text>' ,methods=['GET'])
def Hotel_Sentiment(text):
    if request.method == 'GET':
        if text == text:

           #using aws saved pickle file

       #     s3 = boto3.resource('s3',aws_access_key_id=a, aws_secret_access_key= b ,region_name='us-east-2')
       #     with open('hotelll.pickle', 'wb') as data:
        #        s3.Bucket("projectsss").download_fileobj("hotelll.pickle", data)

        #    with open('hotelll.pickle', 'rb') as data:
        #        resposesum = pickle.load(data)

         #   print(resposesum) 

         #   resposesum=resposesum.predict([text])    
         
         #OR Training online
  
  #stem words
            stemmer = SnowballStemmer('english')

            #stop words removal
            words = stopwords.words("english")

            df=pd.read_csv('https://projectsss.s3.us-east-2.amazonaws.com/hotel-reviewss.csv')

            #regular expression to clean dataset remove space and convert to uppercase
            df['cleaned'] =df['Description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

            #split data
            X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['Is_Response'], test_size=0.2)

            #NLP Trainer
            pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                ('chi',  SelectKBest(chi2, k=10000)),
                                ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

            model = pipeline.fit(X_train, y_train)
            resposesum=model.predict([text6])   
            

            xx = str(resposesum)
           
            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if text == text:
           
                message=yy
               
            return  json.dumps(message)
    
    return  message 

#incase running form command line ,to give full error. and continue running code
if __name__=="__main__":
    app.run(debug=True)



#function for the main app

