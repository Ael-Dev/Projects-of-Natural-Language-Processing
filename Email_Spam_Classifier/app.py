import pickle
import streamlit as st
import re
from nltk.stem import  PorterStemmer
from nltk.corpus import stopwords




# load the preprocessor:
TFIDF = pickle.load(open("./preprocessor.pkl",'rb'))
# load the model:
MODEL = pickle.load(open("./model.pkl",'rb'))


# =================================================================
def preprocessing_text(text_raw, stemmer):
    text_raw = str(text_raw)    
    text = text_raw.lower() # convert all characters in the text string to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.split() # tokenize the cleaned text into words
    text = [word for word in text if word != ' ']
    text = [stemmer.stem(word) for word in text if word not in set(stopwords.words('english'))] # lemmatize each word in the cleaned text
    text = ' '.join(text) # concatenate the cleaned and lemmatized words
    return text # return the cleaned and lemmatized text

stemmer = PorterStemmer() # create a lemmatizer object

# =================================================================
st.title("📧 Email or SMS Spam Classifier 📧")

# Create two columns with relative widths of 1 and 3
col1, col2 = st.columns([1, 3])

# Add a text input box to the first column
text_input = col1.text_area("Enter text here", height=300 )

# Add a button to the second column
if col2.button("Predict"):
    # transform
    # =================================================================
    preprocessed_text = preprocessing_text(text_input,stemmer)
    # vectorize
    # =================================================================
    tfidf_input = TFIDF.transform([preprocessed_text])
    # predict
    result = MODEL.predict(tfidf_input)[0]

     # Display
    if result == 1:
        col2.warning('This is a Spam', icon="⚠️")
    else:
        col2.success('Not Spam')
        





