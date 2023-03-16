import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descargar los datos necesarios para NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Leer el archivo de texto con las respuestas del chat bot
with open (r'C:\Users\aencarnacion\Documents\respuestas.txt', 'r', encoding='utf8', errors='ignore') as file: 
text = file.read()

# Preprocesar el texto
text = text.lower()
sent_tokens = nltk.sent_tokenize(text)
word_tokens = nltk.word_tokenize(text)

# Funciones para preprocesar y lematizar el texto
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Función para obtener la respuesta del chat bot
def get_response(user_input):
    robo_response=''
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='Español')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Lo siento, no te entiendo. ¿Podrías reformular la pregunta?"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

# Iniciar la conversación con el chat bot
print("Hola, soy tu chat bot. ¿En qué puedo ayudarte hoy?")
while True:
    user_input = input()
    user_input = user_input.lower()
    if user_input == 'adios':
        print("Hasta luego.")
        break
    else:
        if user_input.startswith('buscar'):
            print("Estoy buscando la información que necesitas...")
            continue
        else:
            print(get_response(user_input))
