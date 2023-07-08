import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from tensorflow.keras.models import load_model

# converts all words with the same meaning, but different representation to their base form
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# loads training model
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    # Parses a large amount of textual data, divides it up and analyzes individual txt or characters
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Predicts the correct response based on user input
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    # retrieves the response and displays it in the terminal
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    if not result:
        return "I'm sorry, I do not have a response for that"

    return result

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)