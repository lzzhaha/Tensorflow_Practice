import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from process_data import process_data

#define flags

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 0.1
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss


#build and train the word2vec model
def word2vec(batch_gen):
    #define the placeholder for input and output
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

        
    #define weights
    with tf.name_scope('embedded_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                                  name='embed_matrix')
        
    #define hypothesis
    with tf.name_scope('loss'):
        embeded = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
       
        #construct NCE loss variables
        nce_weight = tf.Variable(tf.trucated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                   stddev=1.0/(EMBED_SIZE ** 0.5)),
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
        
        #define loss function to be NCE loss
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                            biases=nce_bias,
                                            labels=target_words,
                                            input=embed,
                                            num_sampled=NUM_SAMPLED,
                                            num_classes=VOCAB_SIZE))
        
        #train optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
        
    sess = tf.Session()
    sess.run(tf.global_varibles_initializer()) 
    
    avg_loss = 0.0
    writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
    
    for index in range(NUM_TRAIN_STEPS):
        centers, targets = next(batch_gen)
        loss_batch, _ = sess.run([loss, optimizer],
                                feed_dict={center_words: centers, target_words: target})
        avg_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
    writer.close()
    
    def main():
        batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
        word2vec(batch_gen)
        
    if __name__ == '__main__':
        main()
