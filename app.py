
#BASIC APP EXAMPLE 
from flask import Flask, render_template
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, EntitiesOptions, KeywordsOptions, SentimentOptions,CategoriesOptions,EmotionOptions
from flask import redirect,request
import pickle

from nltk.sentiment.vader import SentimentIntensityAnalyzer


from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import pandas as pd
import numpy as np



app= Flask(__name__)





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
        username = request.form.get('username')  # access the data inside 
       
        if username == request.form.get('username')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username)

            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            if username == username:
                message=username
                message3=g[0]
        elif username == request.form.get('username'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            

            if username == username:
                message = username
                message2=result2
                
      elif request.form.get("submit_b"):
        # access the data inside 
        username2 = request.form.get('username2')
       
        if username2 == request.form.get('username2')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username2)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if username2 == username2:
                 message=username2
                 message3=g[0]
        elif username2 == request.form.get('username2'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username2,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            

            if username2 == username2:
                message = username2
                message2=result2
                
                
      elif request.form.get("submit_c"):
        # access the data inside 
        username3 = request.form.get('username3')
       
        if username3 == request.form.get('username3')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username3)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if username3 == username3:
                 message=username3
                 message3=g[0]
        elif username3 == request.form.get('username3'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username3,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            

            if username3 == username3:
                message = username3
                message2=result2
                  
      elif request.form.get("submit_d"):
        # access the data inside 
        username4 = request.form.get('username4')
       
        if username4 == request.form.get('username4')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username4)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if username4 == username4:
                message=username4
                message3=g[0]
        elif username4 == request.form.get('username4'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username4,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            

            if username4 == username4:
                message = username4
                message2=result2

      elif request.form.get("submit_e"):        
        username5 = request.form.get('username5')
       
        if username5 == request.form.get('username5')[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username5)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)
            if username5 == username5:
                 message=username5
                 message3=g[0]
        elif username5 == request.form.get('username5'):
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username5,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            resposesum=summarize(username5, ratio=0.5)

          
            print( resposesum)       

            if username5 == username5:
                message = username5
                message2=result2
                message3=json.dumps(resposesum, indent=5)
               
     
      elif request.form.get("submit_f"):        
        username6 = request.form.get('username6')
      
        if username6 == request.form.get('username6'):
           
            
            #using your trained model
            pickle_in = open("hotel2.pickle","rb")
            model = pickle.load(pickle_in)

            resposesum=model.predict([username6])

            print(resposesum)       

            if username6 == username6:
                message = username6
               
                message3=resposesum
     



    return render_template('base.html', message=[message] ,message2= [message3,message2])




#function for the api

@app.route('/home/<string:name>/<int:id>')

def helloworld(name, id):
    return "Hello world" + name + str(id)


@app.route('/Word_Sentiment/<username>' ,methods=['GET'])
def Word_Sentiment(username):
    if request.method == 'GET':
        if username == username[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": username,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if username == username:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            xx = {
            "word": username,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if username == username:
           
                message=yy
               
            return message 
    
    return message
   

@app.route('/Word_Sentiment_and_Emotions/<username>' ,methods=['GET'])
def Word_Sentiment_and_Emotions(username):
    if request.method == 'GET':
        if username == username[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": username,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if username == username:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            xx = {
            "word": username,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if username == username:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Word_Sentiment_and_Category/<username>' ,methods=['GET'])
def Word_Sentiment_and_Category(username):
    if request.method == 'GET':
        if username == username[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": username,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if username == username:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            xx = {
            "word": username,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if username == username:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Word_Sentiment_Emotions_and_Category/<username>' ,methods=['GET'])
def Word_Sentiment_Emotions_and_Category(username):
    if request.method == 'GET':
        if username == username[:10]:

            sid = SentimentIntensityAnalyzer()
        
            response3=sid.polarity_scores(username)
            e=pd.DataFrame(response3,index=[0])

            f=e.apply(lambda score_dict: e['compound'])

            g=e['compound'].apply(func)

            # a Python object (dict):
            x = {
            "word": username,
            
            "Sentiment":g[0]
            }

            # convert into JSON:
            y = json.dumps(x)

            # the result is a JSON string:
            print(y)
            if username == username:
                 message=y
                
        else:
            #using ibm transfer learning model
            authenticator = IAMAuthenticator('rbtGWgXbcWSXelgPEC9Ag-U6ZljN1tXJy4HBl82lSpuE')
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                version='2020-08-01',
                authenticator=authenticator
            )

            natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/a8f00db8-d8f7-465f-9fc6-89722ba0af65')
            
            response = natural_language_understanding.analyze(
                text=username,
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
            
            #using your trained model
            pickle_in = open("nlpsenti.pickle","rb")
            model = pickle.load(pickle_in)

            xx = {
            "word": username,
            
            "Sentiment":result2
            }

            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if username == username:
           
                message=yy
               
            return message 
    
    return message


@app.route('/Hotel_Sentment/<username>' ,methods=['GET'])
def Hotel_Sentiment(username):
    if request.method == 'GET':
        if username == username:

            pickle_in = open("hotel2.pickle","rb")
            model = pickle.load(pickle_in)

            resposesum=model.predict([username])

            print(resposesum) 

            

            xx = dict(np.ndenumerate(resposesum))
           
            # convert into JSON:
            yy = xx
            # the result is a JSON string:
            print(yy)

            if username == username:
           
                message=yy
               
            return message 
    
    return message

#incase running form command line ,to give full error. and continue running code
if __name__=="__main__":
    app.run(debug=True)


#function for the main app

