import tensorflow as tf
import csv

#Transform the value of attribute famhist 
with open('raw_train_data.csv') as raw_file, open('train_data.csv','w') as new_file:
    reader = csv.reader(raw_file, delimiter=',')
    writer = csv.writer(new_file, delimiter=',')
    for row in reader:
        record = row
        
        if(row[4] == 'Present'):
            record[4] = 1
        else:
            record[4] = 0
        
        writer.writerow(record)

with open('raw_test_data.csv') as raw_file, open('test_data.csv','w') as new_file:
    reader = csv.reader(raw_file, delimiter=',')
    writer = csv.writer(new_file, delimiter=',')
    for row in reader:
        record = row
        
        if(row[4] == 'Present'):
            record[4] = 1
        else:
            record[4] = 0
        
        writer.writerow(record)

#Read and decode data from csv file
train_filename_queue = tf.train.string_input_producer(
    ['train_data.csv'], shuffle=False, name='train_file_queue')

test_filename_queue = tf.train.string_input_producer(
    ['test_data.csv'], shuffle=False, name='test_file_queue')
reader = tf.TextLineReader()

key, value = reader.read(train_filename_queue)

record_defaults = [[0.0], [0.0], [0.0],[0.0],[0.0],
                   [0.0],[0.0],[0.0],[0.0],[0.0]]

train_data = tf.decode_csv(value, record_defaults=record_defaults)

key, value = reader.read(test_filename_queue)

test_data = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
    tf.train.batch([train_data[0:-1], train_data[-1]], batch_size=20)

test_features = test_data[0:-1]
test_labels = test_data[-1]

#placeholders for tensors
X = tf.placeholder(tf.float32, shape=[None, 9])

Y = tf.placeholder(tf.float32, shape=[None, 1])


#Training variables
W = tf.Variable(tf.random_normal([9, 1]))

b = tf.Variable(tf.random_normal([1]))


#define hypothesis and cost functions
logit = tf.matmul(X, W) + b

hypothesis = tf.sigmoid(logit)

cost = - tf.reduce_mean((Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis)))


#perform optimization
tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


sess = tf.Session()

sess.run(tf.global_variables_initializer())


#populate the file queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#training process
for step in range(1501):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hypothesis_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    print("at Step: {}, cost: {}".formate(step, cost))

    
#testing process
features = tf.placeholder(tf.float32, shape=[None, 9])
labels = tf.placeholder(tf.float32, shape=[None, 1])
for step in range(67):
    feature, label = sess.run([test_features, test_labels])
    features = sess.run(tf.concat(features, feature))
    labels = sess.run(tf.concat(labels, label))
    
cost = sess.run(cost, feed_dict={X:feature, Y: label})


coord.request_stop()
coord.join(threads)
