import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
# from colab import train_gen
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing

import time
fig = plt.figure()

st.write("""
         # Is it a black vulture? Is it a baltimore oriole? No, it's a Barn Owl!
         """
         )
st.write("This is a simple image classification web app to predict bird species")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                top_classes, top_confidences = predict(image)
                time.sleep(1)
                st.success('Classified')
                for i in range(0, len(top_classes)):
                    st.write(f"{top_classes[i]} with a { top_confidences[i] } % confidence.")

                st.pyplot(fig)



def predict(image):
    train_dir = (r'C:\Users\Andy Zhou\cs1430\cs1430-eagle-vision\cs1430-eagle-vision\data\test')
    data_args = dict(rescale=1./255, validation_split=.20, rotation_range=10, shear_range=5,
                height_shift_range=0.1, width_shift_range=0.1, horizontal_flip=True,
                brightness_range=[0.75, 1.25])

    bag_train = tf.keras.preprocessing.image.ImageDataGenerator(**data_args)
    train_gen = bag_train.flow_from_directory(
        train_dir,
        subset="training",
        shuffle=True,
        target_size=(64, 64))

        

    classifier_model = "colab_model.h5"
    IMAGE_SHAPE = (64,64,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((64,64))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    print("test shape", test_image.shape)
    predictions = model.predict(test_image)
    class_names = train_gen.class_indices
    class_names = dict((v,k) for k,v in class_names.items())
    # print(class_names)
    # print(predictions)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    # result = [f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence.",
    # f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence.",
    # f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence.",]
    print(np.max(scores))
    print(np.argpartition(scores, -3)[-3:])
    top_classes = []
    top_confidences = []
    for i in np.argpartition(scores, -3)[-3:]:
        top_classes.append(class_names[i])
        top_confidences.append((100 * scores[i]).round(2))
    return top_classes, top_confidences
    

if __name__ == "__main__":
    main()


