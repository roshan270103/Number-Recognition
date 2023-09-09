#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[15]:


mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[16]:


X_train, X_test = X_train / 255.0, X_test / 255.0


# In[17]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


# In[18]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[19]:


model.fit(X_train, y_train, epochs=5)


# In[20]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[21]:


predictions = model.predict(X_test)


# In[22]:


first_image = X_test[0]
predicted_digit = np.argmax(predictions[0])
print(f'Predicted digit: {predicted_digit}')


# In[ ]:




