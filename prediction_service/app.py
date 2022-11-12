import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
"""
# deep Classifier project

"""

st.write("Here's our first attempt at using data to create a table:")

model = tf.keras.models.load_model("model.h5")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:

    image = Image.open(uploaded_file)
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # [batch_size, row, col, channel].
    # We are expanding because we do not have the batch size. Expanding adds a dimension along the axis=0.
    result = model.predict(img_array) # [[0.99, 0.01]] since we are using Softmax giving 2 output [cat, dog]
                                      ## If batch size is 2, i.e. 2 images, then O/P [[0.99, 0.01], [0.99, 0.01]] 
    argmax_index = np.argmax(result, axis=1) # If batch size is 2, O/P [0, 0], since 0.99(value at index 0) is more in both rows.
    if argmax_index[0] == 0:
        st.image(image, caption="predicted: cat")
    else:
        st.image(image, caption='predicted: dog')
