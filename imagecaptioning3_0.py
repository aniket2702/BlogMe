

import numpy as np
import keras
import tensorflow as tf
import pickle
import openai
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50
from keras.models import Model, load_model
from keras_preprocessing.sequence import pad_sequences


model = load_model('model_9.h5')
model.make_predict_function()
model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))

model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet.make_predict_function()
def preprocess_image(img):
    img = tf.keras.preprocessing.image.load_img(img, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector

with open('word_to_idx.pkl','rb') as w:
    word_to_idx=pickle.load(w)

with open('idx_to_word.pkl','rb') as w1:
    idx_to_word=pickle.load(w1)

def predict_caption(photo):
    in_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption

def captionimage(image):

    enc=encode_image(image)
    cap= predict_caption(enc)
    return cap

def blogwriter(capt):
    openai.api_key = "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"

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Write a blog on: " + capt + " \n\n",
        temperature=0.6,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']

# c=captionimage('dogs.jpg')
# print(c)
# b=blogwriter(c)
# print(b)
