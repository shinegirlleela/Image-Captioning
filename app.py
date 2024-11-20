import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from tensorflow.keras.models import load_model

# Load the model
model = load_model('image_captioning_model.keras')



max_length = 38


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the image preprocessing function (if you have one)
def preprocess_image(image_path):
    IMG_SIZE = (299, 299)
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Define the caption generation function
def generate_caption(model, tokenizer, image, max_length):
    # Extract features from the input image using the feature extraction model

    inception_model = InceptionV3(weights='imagenet')
    model_incep = Model(inception_model.input, inception_model.layers[-2].output)
    image_features = model_incep.predict(image)

    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Streamlit app
st.title("Image Caption Generator")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image(uploaded_file) # Assuming you have a preprocess_image function

    # Generate caption
    caption = generate_caption(model, tokenizer, image, max_length)

    # Display image and caption
    st.image(uploaded_file, caption=caption, use_column_width=True)