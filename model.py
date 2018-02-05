import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler

data = pd.read_csv('illinois.csv')
data.head()
data.info()
data.shape

dummies = pd.get_dummies(data['Hypertension'],prefix='Hypertension', drop_first=False)
data = pd.concat([data,dummies], axis=1)
data.head()

data = data.drop(['Hypertension'], axis=1)
data.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data[:5,:]

inputs=['% Excessive drinking','Tobacco Consumption','Stroke','Hypertension','Poverty Percent, All Ages','Asthma','COPD']

labels = data['CHD']
# min-max scaling
for each in inputs:
    data[each] = ( data[each] - data[each].min() ) / data[each].max()
    
print(data.head())
print(labels.shape)

features = data.drop(['CHD'], axis=1)
features.head()

features, labels = np.array(features), np.array(labels)
print(len(features), len(labels))

# fraction of examples to keep for training
split_frac = 0.8
n_records = len(features)
split_idx = int(split_frac*n_records)

train_X, train_Y = features[:split_idx], labels[:split_idx]
test_X, test_Y = features[split_idx:], labels[split_idx:]

n_labels= 2
n_features = 10

#hyperparameters

learning_rate = 0.5
n_epochs= 200
n_hidden1 = 200
#batch_size = 128
#display_step = 1

session = tf.Session()

tf.reset_default_graph()
    
with tf.name_scope('inputs'):

    inputs = tf.placeholder(tf.float32,[None, 86], name ='inputs' )
    
with tf.name_scope('target_labels'):
    labels = tf.placeholder(tf.int32, [None,], name='output')
    labels_one_hot = tf.one_hot(labels, 2)

with tf.name_scope('weights'):
    weights = {
        'hidden_layer': tf.Variable(tf.truncated_normal([86,n_hidden1], stddev=0.1), name='hidden_weights'),
        'output':tf.Variable(tf.truncated_normal([n_hidden1, 2], stddev=0.1), name='output_weights')
    }
    
    tf.summary.histogram('hidden_weights', weights['hidden_layer'])
    tf.summary.histogram('output_weights', weights['output'])

with tf.name_scope('biases'):

    bias = {
        'hidden_layer':tf.Variable(tf.zeros([n_hidden1]), name='hidden_biases'),
        'output':tf.Variable(tf.zeros(n_labels), name='output_biases')
    }
    
    tf.summary.histogram('hidden_biases', bias['hidden_layer'])
    tf.summary.histogram('output_biases', bias['output'])
    
with tf.name_scope('hidden_layers'):

    hidden_layer = tf.nn.bias_add(tf.matmul(inputs,weights['hidden_layer']), bias['hidden_layer'])
    hidden_layer = tf.nn.relu(hidden_layer, name='hidden_layer_output')
    
with tf.name_scope('predictions'):

    logits = tf.nn.bias_add(tf.matmul(hidden_layer, weights['output']), bias['output'], name='logits')
    pred = tf.nn.softmax(logits, name='predictions')
    tf.summary.histogram('predictions', pred)

with tf.name_scope('cost'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot, name='cross_entropy')
    cost = tf.reduce_mean(entropy, name='cost')
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    #tensorboard
    train_writer = tf.summary.FileWriter('./logs/3', sess.graph)
    
    for epoch in range(n_epochs):
        
        summary,_, loss = sess.run([merged,optimizer, cost], feed_dict={inputs:train_X, labels:train_Y})
       
        print("Epoch: {0} ; training loss: {1}".format(epoch, loss))
        
        train_writer.add_summary(summary, epoch+1)
        
    print('training finished')
    
     # testing the model on test data
        
#         test_loss,logits = sess.run([loss,logits],feed_dict={inputs:test_X,labels:test_Y})
    
#         predictions = tf.nn.softmax(logits)
    
#         correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf.one_hot(test_Y), 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) 
    
#         print('model accuracy : {}'.format(accuracy))
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({inputs: test_X, labels: test_Y}))

    def predict(example):
        return sess.run(pred,feed_dict={inputs:example})

    print(predict([test_X[0]]))