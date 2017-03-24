'''
LSTM code for sleep classification
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#import BN_LSTMCell as BNLSTM
from sklearn.cross_validation import train_test_split
from sklearn.metrics import cohen_kappa_score
from tensorflow.core.protobuf import saver_pb2


# Parameters
fp = open('model_results_lstm_v1_2.txt','wb')
learning_rate = 0.00015
training_iters = 8000000
training_iters = 7000000
batch_size = 300
display_step = 100

# Network Parameters
n_input = 96  
n_steps = 20  
# n_hidden = 200 
n_hidden = 1000 
n_classes = 5

strw =  'learning rate' + str(learning_rate) +',n_hidden, ' + str(n_hidden) + '\n' 
fp.write(strw)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes],stddev=0.35))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = tf.nn.rnn_cell._linear([inputs, h], 4 * self._num_units, False)

            i, j, f, o = tf.split(1, 4, concat)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f = ln(f, scope='f/')
            o = ln(o, scope='o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow

    ## for LSTM cell initialization 
    #cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # for GRU cell initialization 
    #cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)

    # for Layer normalized cell initialization 
    #cell = LayerNormalizedLSTMCell(n_hidden)
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    #cell = BNLSTM.BN_LSTMCell(n_hidden, is_training = True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.9)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 5, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    #outputs, states = rnn.rnn(cell, x, dtype = tf.float32) 
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

model_path = "./model.ckpt"

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    ct = 0
    acc_mat = []
    test_acc_mat = []

#    rnn_20  = np.load('./data/rnn_100_cases_new.npy')
    # rnn_20  = np.load('./data/rnn_100_cases_18th_Twin.npy')

    #rnn_20  = np.load('./data/rnn_100_cases_17th.npy')
    rnn_20 = np.load('rnn_1200_cases_Twin.npy')   
    sleep_stages_mat  =  rnn_20[()]['sleep_stages']
    #feature_mat  = np.load('./data/final_model.npy')
    feature_mat = np.load('new_final_data_18th_Twin_2.npy')
    sleep_stages_mat = sleep_stages_mat[100:500000]
    print('Training data set shape:',feature_mat.shape, sleep_stages_mat.shape)

    feature_mat, features_test , sleep_stages_mat, sleep_stages_test = train_test_split( feature_mat, sleep_stages_mat, test_size=0.2, random_state=42)
    #rnn_test = np.load('./data/rnn_20_cases_test.npy') 
    #fMAT, ss = rnn_test[()]['feature'], rnn_test[()]['sleep_stages']
#    print('Testing data set shape:',fMAT.shape, ss.shape)

#    features_test = fMAT[:]
#    sleep_stages_test = ss[:]
    batch_x_test = features_test.reshape(features_test.shape[0],20,96)
    batch_y_test = np.zeros((sleep_stages_test.shape[0], 5)) 
    
    # only using 100 cases of test dataset
    print(batch_x_test.shape, batch_y_test.shape)
    for v in range(sleep_stages_test.shape[0]):
        batch_y_test[v, int(sleep_stages_test[v])-1] = 1.0
    batch_x_test = batch_x_test.reshape((features_test.shape[0], n_steps, n_input))
    print(batch_x_test.shape)
    batch_x_test1 = batch_x_test[1:10000,:,:]
    batch_y_test1 = batch_y_test[1:10000,:]

    batch_x_test2 = batch_x_test[10000:20000,:,:]
    batch_y_test2 = batch_y_test[10000:20000,:]

    
    while step * batch_size < training_iters:
        try:
            ct = ct + batch_size
            start_ind, end_ind = ct, ct+ batch_size
            features = feature_mat[start_ind:end_ind]
            sleep_stages = sleep_stages_mat[start_ind:end_ind]
            


            batch_x = features.reshape(features.shape[0],20,96)
            batch_y = np.zeros((sleep_stages.shape[0], 5))            
            for v in range(sleep_stages.shape[0]):
                batch_y[v, int(sleep_stages[v])-1] = 1.0


            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:
                # Calculate batch accuracy
                
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                acc_mat.append(acc)

                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})


                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            if step % display_step == 0:
                test_acc = sess.run(accuracy, feed_dict = {x: batch_x_test1, y: batch_y_test1})
                y_pred     = sess.run(pred,     feed_dict = {x: batch_x_test1, y: batch_y_test1})
                
                pred_inds = np.argmax(y_pred, axis=1)
                true_inds = np.argmax(batch_y_test1,axis=1)
                kappa = cohen_kappa_score(pred_inds, true_inds)

                print('Testing Accuracy is:', test_acc,'Kappa value is:',kappa)

                test_acc = sess.run(accuracy, feed_dict = {x: batch_x_test2, y: batch_y_test2})
                y_pred     = sess.run(pred,     feed_dict = {x: batch_x_test2, y: batch_y_test2})
                
                pred_inds = np.argmax(y_pred, axis=1)
                true_inds = np.argmax(batch_y_test2,axis=1)
                kappa = cohen_kappa_score(pred_inds, true_inds)
                print('Testing Accuracy is:', test_acc,'Kappa value is:',kappa)



                str_w = str(step) +','+ str(test_acc) + str(kappa) +', test_acc, \n'
                fp.write(str_w)
		#if kappa>0.70:
            save_path = saver.save(sess, model_path)
            step += 1

        except:
            print('could not train')
            ct =0

fp.close()



