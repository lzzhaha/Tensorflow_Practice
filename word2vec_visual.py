from __future__ import absolute_import
from __future__ import division
from __future__ import print_functioni
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from process_data import process_data
import utils

#define flags
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000

class SkipGramModel:
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placehodler(self):
        with tf.name_scope('data'):
            self.center_words = tf.place_holder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.place_holder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('embed'):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                self.embed_size], -1.0, 1.0),
                                                                name='embed_matrix')

    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('loss'):
                embed = tf.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

                nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], 
                                                            stddev=1.0/(self.embed_size ** 0.5)),
                                                            name='nce_weight')
                
                nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')
                
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                            biases=nce_bias,
                                            labels=self.target_words,
                                            input=embed,
                                            num_sampled=self.num_sampled,
                                            num_classes=self.vocab_size), name='loss')
            
    
    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)

    def _build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver()

    intial_step = 0
    utils.make_dir('check_points')
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    check_pt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    avg_loss = 0.0
    writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
    intial_step = model.global_step.eval()
    for index in range(intial_step, initial_step + num_train_steps):
        centers, targets = next(batch_gen)
        
        loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                        feed_dict={model.ceter_words: centers, model.target_words: targets})
        writer.add_summary(summary, global_step=index)
        avg_loss += loss_batch
        if(index + 1)%  SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/skip-gram', index)


def main():
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_grap()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == '__main__':
    main()
