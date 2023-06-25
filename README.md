# AI_chatbot

## What is NLP?
> NLP **(Natural Language Processing)** is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.

```
import nltk
import random
import string
import warnings
from datetime import date
from datetime import datetime

warnings.filterwarnings('ignore')

f = open(".\\file1.txt", 'r' , errors='ignore')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) #converts to list of sentences
word_tokens = nltk.word_tokenize(raw) #converts to list of words

sentToken = sent_tokens[:4]
#print(sentToken)
wordToken = word_tokens[:4]
#print(wordToken)

#prepocesssing
#Lemmatization -> It is a process used to create meaningful sentences

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens): #tokens-> user inputs
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#greetings

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)  

#vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatbot_response=chatbot_response+"I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response+sent_tokens[idx]
        return chatbot_response
 
if __name__== "__main__":
    flag=True
    print("ROBO: My name is Robo. If you want to exit, type Bye!")
    while(flag==True):
        user_response = input()
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print("ROBO: You are welcome..")
            else:
                if(greeting(user_response)!=None):
                    print("ROBO: "+greeting(user_response))
                elif("date" in user_response):
                    today = date.today()
                    print("ROBO: Today's date: " , today)
                elif("time" in user_response):
                    now = datetime.now()
                    dt_string = now.strftime("%H:%M:%S")
                    print("ROBO: Current time: " , dt_string)
                elif("how are you" in user_response):
                    print("ROBO: I,m great!")   
                else:
                    print("ROBO: ",end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:   
            flag=False
            print("ROBO: Bye! take care..") 
```

