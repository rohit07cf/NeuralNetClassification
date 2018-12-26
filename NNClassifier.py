
# coding: utf-8

# In[1]:


import tensorflow as tf;
import sklearn.datasets;
import pandas as pd
import numpy as np


# In[2]:


iris_ds = sklearn.datasets.load_iris(return_X_y=False)


# In[3]:


iris_ds.target


# In[4]:


iris_data = pd.DataFrame(data=iris_ds.data,columns=iris_ds.feature_names)


# In[5]:


iris_data.head()


# In[6]:


from sklearn.model_selection import train_test_split
import sklearn


# In[7]:


min_max_scaler = sklearn.preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(iris_data)


# In[8]:


from sklearn.preprocessing import OneHotEncoder


# In[9]:


encoder = OneHotEncoder(3)
label = encoder.fit_transform(iris_ds.target.reshape(-1,1))
label = label.todense()


# In[10]:


trainx,testx,trainy,testy = train_test_split(scaled_data,label)


# In[11]:


print (trainx.shape)


# In[12]:


x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])
w1 = tf.get_variable(name='w1',dtype=tf.float32,shape=[5,4])
w2 = tf.get_variable(name='w2',dtype=tf.float32,shape=(3,5))
b1 = tf.get_variable(name='b1',dtype=tf.float32,shape=(5,1))
b2 = tf.get_variable(name='b2',dtype=tf.float32,shape=(3,1))


# In[13]:


from tensorflow.contrib.keras import layers

def build_model(x):
    y1 = layers.Dense(units=5,activation=tf.nn.relu, input_shape=(None,4))(x)
    print (y1.shape)
    y2 = layers.Dense(units=3,activation=tf.nn.softmax,input_shape=(None,5))(y1)
    print (y2.shape)
    return y2


# In[14]:


# Applying build_model
y_hat = build_model(x)


# In[15]:


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1)),tf.float64))


# In[16]:


entropy_loss = tf.losses.softmax_cross_entropy(logits=y_hat,onehot_labels=y)
optimizer = tf.train.AdamOptimizer().minimize(entropy_loss)


# In[17]:


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for j in range(300):
        for i in range(len(trainx)):
            x_val = np.array(trainx[i].reshape(1,-1),dtype=np.float32)
            y_val = np.array(trainy[i].reshape(1,-1),dtype=np.float32)
            optimizer.run(feed_dict={x:x_val,y:y_val})
            _loss, = sess.run([entropy_loss] ,feed_dict={x:x_val,y:y_val})
        if(j%5==0):
            _acc = sess.run(accuracy,feed_dict={x:testx,y:testy})
            print ('loss=%s,accuracy=%s'%(_loss,_acc))

