# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:00:52 2020

@author: sharm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:18:54 2020

@author: sharm
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Reading data files for rating,users and books
book_rating_data_set = pd.read_csv('C:/Users/sharm/Downloads/BX-CSV-Dump (1)/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user_data_set = pd.read_csv('C:/Users/sharm/Downloads/BX-CSV-Dump (1)/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book_data_set = pd.read_csv('C:/Users/sharm/Downloads/BX-CSV-Dump (1)/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")

# Data cleaning 
# combining ratings file with book file on keeping ISBN and then dropping cols which are not required for further calculation
book_rating_file = pd.merge(book_rating_data_set, book_data_set, on='ISBN')
cols = ['Year-Of-Publication', 'Publisher', 'Book-Author', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
book_rating_file.drop(cols, axis=1, inplace=True)
print(book_rating_file.head(10))

# extracting information for finding ratings count for each book based on title and rating by each user
rating_counting = (book_rating_file.
     groupby(by = ['Book-Title'])['Book-Rating'].
     count().
     reset_index().
     rename(columns = {'Book-Rating': 'RatingCount_book'})
     [['Book-Title', 'RatingCount_book']]
    )
print(rating_counting.head(10))

# Finding out the counting of ratings for the threshold value to find out ratings for each book
threshold = 25
rating_counting = rating_counting.query('RatingCount_book >= @threshold')
print(rating_counting.head(10))


# combining above counted ratings with book rating data set for each book title and extracting the results from rating dataframe
user_rating = pd.merge(rating_counting, book_rating_file, left_on='Book-Title', right_on='Book-Title', how='left')
print(user_rating.head(10))

# Finding out user counts for each book rating  for the users
user_counting = (user_rating.
     groupby(by = ['User-ID'])['Book-Rating'].
     count().
     reset_index().
     rename(columns = {'Book-Rating': 'RatingCount_user'})
     [['User-ID', 'RatingCount_user']]
    )
print(user_counting.head(10))


# finding out common results based on user id for each user 
combined_results = user_rating.merge(user_counting, left_on = 'User-ID', right_on = 'User-ID', how = 'inner')
print(combined_results.head(10))


# processing and scaling the combined results for the book ratings
scaler = MinMaxScaler()
combined_results['Book-Rating'] = combined_results['Book-Rating'].values.astype(float)
rating_scaled = pd.DataFrame(scaler.fit_transform(combined_results['Book-Rating'].values.reshape(-1,1)))
combined_results['Book-Rating'] = rating_scaled


# dropping the duplicate results
combined_results = combined_results.drop_duplicates(['User-ID', 'Book-Title'])
user_book_matrix = combined_results.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
user_book_matrix.fillna(0, inplace=True)
users = user_book_matrix.index.tolist()
books = user_book_matrix.columns.tolist()


# Now implementing tensorflow on the above cleaned combined data results 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



# defining the parameters for building neural network from scratch
num_input = combined_results['Book-Title'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])
# calcuating weights and biases for the layers 
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}

# defining encoders and decoders
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

# invoking encoders and decoders functions
encoder_output = encoder(X)
decoder_output = decoder(encoder_output)


# predicting valuesbased on the decoder output
y_pred = decoder_output
y_true = X


# calculating loss values and optimizing the results  and then predicting the data
loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()


#iterating through data for calculating the predicting values for each epoch
with tf.Session() as session:
    epochs = 100
    batch_size = 35
    session.run(init)
    session.run(local_init)
    num_batches = int(user_book_matrix.shape[0] / batch_size)
    user_book_matrix = np.array_split(user_book_matrix, num_batches)
    
    for i in range(epochs):
        avg_cost = 0
        for batch in user_book_matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("epoch: {} Loss: {}".format(i + 1, avg_cost))
        
        
        
        
# doing calculations on predicted data for users book rating 
user_book_matrix = np.concatenate(user_book_matrix, axis=0)
predictions = session.run(decoder_output, feed_dict={X: user_book_matrix})
predicted_data = predictions.append(pd.DataFrame(predictions))
predicted_data = predicted_data.stack().reset_index(name='Book-Rating')
predicted_data.columns = ['User-ID', 'Book-Title', 'Book-Rating']
predicted_data['User-ID'] = predicted_data['User-ID'].map(lambda value: users[value])
predicted_data['Book-Title'] = predicted_data['Book-Title'].map(lambda value: books[value])
keys = ['User-ID', 'Book-Title']
index_first_column = predicted_data.set_index(keys).index
index_second_column = combined_results.set_index(keys).index


# finding out the top ranked results from the predicted values
top_ranked = predicted_data[~index_first_column.isin(index_second_column)]
top_ranked = top_ranked.sort_values(['User-ID', 'Book-Rating'], ascending=[True, False])
top_ranked = top_ranked.groupby('User-ID').head(10)
print(top_ranked)



# testing results for one of the record
print(top_ranked.loc[top_ranked['User-ID'] == 180187])
print(book_rating_data_set.loc[book_rating_data_set['User-ID'] == 180187].sort_values(by=['Book-Rating'], ascending=False))

