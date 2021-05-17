import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import string
from PIL import Image
import os
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from IPython.display import Image as ima, display
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from tqdm import tqdm_notebook as tqdm
import gtts  
from playsound import playsound
tqdm().pandas()

def main():
    #try:
      st.title("Image Captioning through Deep Learning (ResNet50 architecture)")
      st.sidebar.title("Image Captioning through Deep Learning (ResNet50 architecture)")
      st.markdown("Please fill up certain information present on left-side bar")
      st.sidebar.markdown("Let's get started!")
      def extract_features(filename, model):
              #try:
              image = Image.open(filename)
              #except:
                  #print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
              
              image = image.resize((299,299))
              image = np.array(image)
              
              if image.shape[2] == 4: 
                  image = image[..., :3]
              image = np.expand_dims(image, axis=0)
              image = image/127.5
              image = image - 1.0
              feature = model.predict(image)
              return feature

      def word_for_id(integer, tokenizer):
          for word, index in tokenizer.word_index.items():
              if index == integer:
                return word
              #return None

      def generate_desc(model, tokenizer, photo, max_length):
          in_text = 'start'
          for i in range(max_length):
              sequence = tokenizer.texts_to_sequences([in_text])[0]
              sequence = pad_sequences([sequence], maxlen=max_length)
              pred = model.predict([photo,sequence], verbose=0)
              pred = np.argmax(pred)
              word = word_for_id(pred, tokenizer)
              if word is None:
                  break
              in_text += ' ' + word
              if word == 'end':
                  break
          return in_text
      def corr(x):
        corr, _ = pearsonr(x, y[0])
        return corr
      def getspeech(x):
        t1 = gtts.gTTS(x)
        t1.save("file.mp3")
        #playsound("file.mp3")
      
      choice = st.radio("Select a choice",("Caption-Generation","Similar image retrieval using Pearson's Correlation","Text-To-Speech conversion"))
      if choice == "Caption-Generation":
        st.sidebar.subheader("Please upload an image file")
        img_path = st.sidebar.file_uploader("Upload an image",key='train')
        #img_path = '/content/drive/MyDrive/test images/test_image.jpg'
        #img_path = '/content/drive/MyDrive/Flickr8k_Dataset/Flicker8k_Dataset/3482062809_3b694322c4.jpg'
        #img_path = '/content/drive/MyDrive/test images/test_image1.jpg'
        max_length = 32
        tokenizer = load(open("/content/drive/MyDrive/Frontend/tokenizer.p","rb"))
        #model = load_model('models/model_9.h5')
        model = load_model('/content/drive/MyDrive/Frontend/models/model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")
        cap = st.sidebar.button('Generate-Caption',key='cap')
        if cap:
          photo = extract_features(img_path, xception_model)
          img = Image.open(img_path)
          description = generate_desc(model, tokenizer, photo, max_length)
          print("\n\n")
          description = description.replace("start","")
          description = description.replace("end","")
          getspeech(description)
          col1, col2, col3 = st.beta_columns([1,6,1])

          with col1:
           st.write("")

          with col2:
           st.image(img)
           st.write("Caption-Generated: **",description.title(),"**")
          with col3:
           st.write("")
                  #st.image(img)
                  #st.write("Caption-Generated: ",description)
        
      elif choice == "Similar image retrieval using Pearson's Correlation":
            images_path = '/content/drive/MyDrive/Flickr8k_Dataset/Flicker8k_Dataset'
            pd_dataset = pd.read_csv("/content/drive/MyDrive/Frontend/flickr_8k_train_dataset.txt", delimiter='\t')
            tfidf = TfidfVectorizer(sublinear_tf= True, 
                       min_df = 5, 
                       norm= 'l2', 
                       ngram_range= (1,3), 
                       stop_words ='english',
                      max_features=1500)
            x_transform = pd_dataset['captions']
            features = tfidf.fit_transform(x_transform).toarray()
            input = st.sidebar.text_input("Enter Input Sentence")
            input = "<start> "+input+" <end>"
            input = [input]
            st.write(input)
            pd_dataset['features']= list(tfidf.fit_transform(x_transform).toarray())
            y=tfidf.transform(input).toarray()
            pd_dataset['corr'] = pd_dataset['features'].apply(corr)
            pd_dataset.drop(['features'], axis=1, inplace=True)
            fetc = st.sidebar.button('Fetch Images',key='fetc')
            if fetc:
                  df=pd_dataset.sort_values(by=['corr'],ascending=False)
                  st.write(df)
                  for i,j in df.iloc[0:5].iterrows():
                        #q = Image(filename=images_path+"/"+j[0])
                        st.image(images_path+"/"+j[0])
                        st.write(j[1])
        
      else:
         #sound = st.sidebar.button('Play',key='sound')
         #if sound: 
            #getspeech(description)
            audio_file = open("file.mp3",'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg',start_time=0)
    #except:
      #pass      
if __name__ == '__main__':
  main()
