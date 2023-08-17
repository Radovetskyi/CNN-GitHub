import streamlit as st
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageOps
# import cv2

def load_model():
    # model = keras.models.load_model('/Users/home/Desktop/CNN_Prjct/final_model_2.h5')
    model = keras.models.load_model('/app/final_model_2.h5')
    return model

def load_picture():
    image = st.file_uploader(label='Upload your image! (Airplane, Auto, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)', type=['.jpeg', '.jpg', '.png'])
    return image

def predict(model, img):
    pil_image = Image.open(img)
    pil_image = pil_image.resize((32, 32))
    x = img_to_array(pil_image)
    x /= 255
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    return pred

dict_ = {0: 'Airplane', 1: 'Auto', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}
columns = list(dict_.values())

# def import_and_predict(image_data, model):
#         size = (180, 180)    
#         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#         image = np.asarray(image)
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_reshape = img[np.newaxis, ...]
#         prediction = model.predict(img_reshape)
#         return prediction

def main():

    st.title('Image Classification (CIFAR-10)')

    with st.spinner('Model is being loaded..'):
        model=load_model()
    left, right = st.columns(2)

    with left:
        img = load_picture()

    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    with right:
        if img is None:
            st.text("Please upload an image file")
        else:
            classes = predict(model, img=img)
            for cl in classes:
                class_index = np.argmax(cl)
                class_name = dict_.get(class_index)
                st.write('Predicton ------> ',class_name)
            if img is None:
                pass
            else:
                st.image(img)
    if img is None:
        pass
    else:
        classes = predict(model, img=img)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Побудова графіка
        ax.bar(columns, classes[0])  # Перший рядок з 'classes' містить вірогідності для одного прикладу

        # Додаткові налаштування графіка
        ax.set_ylabel('Probability')
        ax.set_xlabel('Classes')
        ax.set_title('Predicted Probabilities for Each Class')
        ax.set_xticklabels(columns, rotation=45, ha='right')

        # Показ графіка
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)  

if __name__ == '__main__':
    main()