from flask import Flask,render_template
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
import requests
import pandas as pd
from patsy import dmatrices

app=Flask(__name__)





# ---------------------------------------------------------------------

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('./Statics/Datas/vectorizer.pkl','rb'))
model = pickle.load(open('./Statics/Datas/model.pkl','rb'))



# ---------------------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classifi',methods= ['GET','POST'])
def priduction():
    status=False
    if request.method=="POST":
        try:
            if request.form:
                k=''
                message=request.form['myText']
                print(message)
                transformed_sms = transform_text(message)
                # 2. vectorize
                vector_input = tfidf.transform([transformed_sms])
                # 3. predict
                result = model.predict(vector_input)[0]
                # 4. Display
                if result == 1:
                    k="Spam"
                    
                else:
                    k="Not Spam"
                status=True

                return render_template('index.html',email=k,status=status,message=message)
                            
                
                
                

        except Exception as e:
            error={'error':e}
            print(e)
            return render_template('index.html')
    
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)