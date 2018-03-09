import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
from model_utils import data_utils

finetune = True
finetune_epochs = 50


ignore_idx = 89

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')


def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    return tf.nn.elu(tf.add(conv2d(x, weight, kernel_stride), bias))


def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')


def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))


def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)


#######################################
## dataset paras
#######################################


dataset_dir = './processed/3D_data/'

conv_1_shape = '3*3*1*32'
pool_1_shape = 'None'

conv_2_shape = '3*3*1*64'
pool_2_shape = 'None'

conv_3_shape = '3*3*1*128'
pool_3_shape = 'None'

conv_4_shape = 'None'
pool_4_shape = 'None'


window_size = 10
n_lstm_layers = 2
fc_size = 1024
n_fc_in = 1024
n_fc_out = 1024

dropout_prob = 0.5
calibration = 'N'
norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = False




input_channel_num = 1

input_height = 10
input_width = 11

n_labels = 4

# training parameter
lambda_loss_amount = 0.0005
training_epochs = 100

batch_size = 200

accuracy_batch_size = 200
kernel_height_1st = 3
kernel_width_1st = 3

kernel_height_2nd = 3
kernel_width_2nd = 3

kernel_height_3rd = 3
kernel_width_3rd = 3

kernel_stride = 1
conv_channel_num = 32
# pooling parameter
pooling_height = 2
pooling_width = 2

pooling_stride = 2

# algorithn parameter
learning_rate = 1e-4



# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='X')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
phase_train = tf.placeholder(tf.bool, name = 'phase_train')

# first CNN layer
conv_1 = apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
# pool_1 = apply_max_pooling(conv_1, pooling_height, pooling_width, pooling_stride)
# second CNN layer
conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride)
# pool_2 = apply_max_pooling(conv_2, pooling_height, pooling_width, pooling_stride)
# third CNN layer
conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride)
# fully connected layer
shape = conv_3.get_shape().as_list()

pool_2_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]])
fc = apply_fully_connect(pool_2_flat, shape[1]*shape[2]*shape[3], fc_size)

# dropout regularizer
# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing

fc_drop = tf.nn.dropout(fc, keep_prob)

# fc_drop size [batch_size*window_size, fc_size]
# lstm_in size [batch_size, window_size, fc_size]
lstm_in = tf.reshape(fc_drop, [-1, window_size, fc_size])

cells = []
for _ in range(n_lstm_layers):
    cell = tf.contrib.rnn.BasicLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)


    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# output ==> [batch, step, n_fc_in]
output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)


output = tf.unstack(tf.transpose(output, [1, 0, 2]), name = 'lstm_out')
rnn_output = output[-1]
shape_rnn_out = rnn_output.get_shape().as_list()
# fc_out ==> [batch_size, n_fc_out]
fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

# keep_prob = tf.placeholder(tf.float32)
fc_drop = tf.nn.dropout(fc_out, keep_prob)

# readout layer
y_ = apply_readout(fc_drop, shape_rnn_out[1], n_labels)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
output_dir = './results/'
output_file = output_dir+'cnn_rnn.txt'


selected_idx = np.array(range(1, 110))
np.random.shuffle(selected_idx)
selected_idx = selected_idx[:20]

test_loss_history = np.empty(shape=[0], dtype=float)
test_accuracy_history = np.empty(shape=[0], dtype=float)

for test_subject in selected_idx:
    if test_subject == ignore_idx:
        continue

    if finetune:
        pretrain_x, pretrain_y, finetune_x, finetune_y, test_x, test_y = data_utils.leave_one_finetune_split(
            test_subject, split=True)
        train_x, train_y = pretrain_x, pretrain_y

    else:
        train_x, train_y, test_x, test_y = data_utils.leave_one_finetune_split(test_subject)

    idx = np.array(range(len(train_y)))
    train_x = train_x[idx]
    train_y = train_y[idx]
    print('Working on subject ', format(test_subject, '03d'))

    train_x = train_x.reshape(len(train_x), window_size, 10, 11, 1)
    test_x = test_x.reshape(len(test_x), window_size, 10, 11, 1)
    train_y = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
    test_y = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

    batch_num_per_epoch = train_x.shape[0]//batch_size
    train_accuracy_batchNB = train_x.shape[0] // accuracy_batch_size
    test_accuracy_batchNB = test_x.shape[0] // accuracy_batch_size

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        early_stopping_count = 0

        train_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)

        for epoch in range(training_epochs):

            cost_history = np.zeros(shape=[0], dtype=float)
            for batch in range(batch_num_per_epoch):
                offset = (batch * batch_size) % (train_y.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
                batch_x = batch_x.reshape(len(batch_x) * window_size, 10, 11, 1)
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = session.run([optimizer, cost],
                                   feed_dict={X: batch_x, Y: batch_y, keep_prob: 1 - dropout_prob, phase_train: True})
                cost_history = np.append(cost_history, c)

            train_accuracy = np.zeros(shape=[0], dtype=float)
            test_accuracy  = np.zeros(shape=[0], dtype=float)
            test_loss      = np.zeros(shape=[0], dtype=float)
            train_loss     = np.zeros(shape=[0], dtype=float)

            for i in range(train_accuracy_batchNB):
                offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size)
                train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                train_batch_x = train_batch_x.reshape(len(train_batch_x) * window_size, 10, 11, 1)
                train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]

                train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, phase_train: False})
                train_loss = np.append(train_loss, train_c)
                train_accuracy = np.append(train_accuracy, train_a)
            print("Epoch: ", epoch + 1, " Training Cost: ", np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
            train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
            train_loss_save     = np.append(train_loss_save, np.mean(train_loss))

            for j in range(test_accuracy_batchNB):
                offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                test_batch_x = test_batch_x.reshape(len(test_batch_x) * window_size, 10, 11, 1)
                test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]

                test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})

                test_accuracy = np.append(test_accuracy, test_a)
                test_loss = np.append(test_loss, test_c)

            print("Epoch: ", epoch + 1, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
            test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
            test_loss_save = np.append(test_loss_save, np.mean(test_loss))
            if epoch > 0:
                if train_loss_save[epoch] < train_loss_save[epoch - 1]:
                    if test_loss_save[epoch] > test_loss_save[epoch - 1]:
                        early_stopping_count += 1
                    else:
                        early_stopping_count = 0
                if early_stopping_count > 3:
                    print('Early stopping.')
                    break

        with open(output_file, 'a') as ref:
            print('Subject id:', format(test_subject, '03d'), file=ref)
            print('Final training loss:', np.mean(train_loss), "\tFinal Training Accuracy:", np.mean(train_accuracy),
                  file=ref)
        if finetune:
            print("---------------------------------------------Finetuning--------------------------------------------------:")
            print('Samples used for finetuning:', len(finetune_y))
            finetune_x = finetune_x.reshape(len(finetune_x), window_size, 10, 11, 1)
            finetune_y = np.asarray(pd.get_dummies(finetune_y), dtype=np.int8)

            finetune_accuracy_save = np.zeros(shape=[0], dtype=float)
            finetine_loss_save = np.zeros(shape=[0], dtype=float)

            finetune_batch_per_epoch = len(finetune_y) // batch_size

            early_stopping_count = 0
            last_test_loss = np.mean(test_loss)

            for epoch in range(finetune_epochs):
                idx = np.array(range(len(finetune_y)))
                finetune_x = finetune_x[idx]
                finetune_y = finetune_y[idx]

                finetune_cost_history = np.zeros(shape=[0], dtype=float)
                for batch in range(finetune_batch_per_epoch):
                    offset = (batch * batch_size) % (finetune_y.shape[0] - batch_size)
                    batch_x = finetune_x[offset:(offset + batch_size), :, :, :, :]
                    batch_x = batch_x.reshape(len(batch_x) * window_size, 10, 11, 1)
                    batch_y = finetune_y[offset:(offset + batch_size), :]
                    _, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y,  keep_prob: 1 - dropout_prob})
                    cost_history = np.append(finetune_cost_history, c)
                finetune_accuracy = np.zeros(shape=[0], dtype=float)
                test_accuracy = np.zeros(shape=[0], dtype=float)
                finetune_loss = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)

                for i in range(finetune_batch_per_epoch):
                    offset = (i * accuracy_batch_size) % (finetune_y.shape[0] - accuracy_batch_size)
                    train_batch_x = finetune_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                    train_batch_x = train_batch_x.reshape(len(train_batch_x) * window_size, 10, 11, 1)
                    train_batch_y = finetune_y[offset:(offset + accuracy_batch_size), :]
                    train_a, train_c = session.run([accuracy, cost],
                                                   feed_dict={X: train_batch_x, Y: train_batch_y,
                                                              keep_prob: 1.0})
                    train_loss = np.append(train_loss, train_c)
                    train_accuracy = np.append(train_accuracy, train_a)
                print('Epoch: ', epoch + 1, "\nFinetuning Cost:", np.mean(train_loss), "Finetuning Accuracy:",
                      np.mean(train_accuracy))
                finetune_accuracy_save = np.append(finetune_accuracy_save, np.mean(train_accuracy))
                finetine_loss_save = np.append(finetine_loss_save, np.mean(train_loss))

                for i in range(test_accuracy_batchNB):
                    offset = (i * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                    test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                    test_batch_x = test_batch_x.reshape(len(test_batch_x) * window_size, 10, 11, 1)
                    test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
                    test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})
                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)
                print('Finetuning Test Cost:', np.mean(test_loss), "Finetuning Test Accuracy:", np.mean(test_accuracy), '\n')

                if epoch > 0:
                    if train_loss_save[epoch] < train_loss_save[epoch - 1]:
                        if test_loss_save[epoch] > test_loss_save[epoch - 1]:
                            early_stopping_count += 1
                        else:
                            early_stopping_count = 0
                    if early_stopping_count > 3:
                        print('Early stopping.\nFinetuning stops.')
                        break

                test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                test_loss_save = np.append(test_loss_save, np.mean(test_loss))

        with open(output_file, 'a') as ref:
            print('Final fintuning loss:', np.mean(train_loss), "\tFinal fintuning Accuracy:", np.mean(train_accuracy), file=ref)
        test_accuracy = np.zeros(shape=[0], dtype=float)
        test_loss = np.zeros(shape=[0], dtype=float)
        # test_pred = np.zeros(shape=[0], dtype=float)
        # test_true = np.zeros(shape=[0, 5], dtype=float)
        # test_posi = np.zeros(shape=[0, 5], dtype=float)
        for k in range(test_accuracy_batchNB):
            offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
            test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
            test_batch_x = test_batch_x.reshape(len(test_batch_x) * window_size, 10, 11, 1)
            test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
            test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_accuracy = np.append(test_accuracy, test_a)
            test_loss = np.append(test_loss, test_c)
            # test_pred = np.append(test_pred, test_p)
            # test_true = np.vstack([test_true, test_t])
            # test_posi = np.vstack([test_posi, test_r])
        # # test_true = tf.argmax(test_true, 1)
        # test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
        # test_true_list = tf.argmax(test_true, 1).eval()
        # print(test_pred.shape)
        # print(test_pred)
        # print(test_true.shape)
        # print(test_true)
        #
        # # recall
        # test_recall = recall_score(test_true, test_pred_1_hot, average=None)
        # # precision
        # test_precision = precision_score(test_true, test_pred_1_hot, average=None)
        # # f1 score
        # test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
        # # auc
        # test_auc = roc_auc_score(test_true, test_pred_1_hot, average=None)
        # # confusion matrix
        # confusion_matrix = confusion_matrix(test_true_list, test_pred)
        #
        # print("********************recall:", test_recall)
        # print("*****************precision:", test_precision)
        # print("******************test_auc:", test_auc)
        # print("******************f1_score:", test_f1)
        # print("**********confusion_matrix:\n", confusion_matrix)

        print("(" + time.asctime(time.localtime(time.time())) + ')\nFinal Test Cost:', np.mean(test_loss), 'Final Test Accuracy:', np.mean(test_accuracy), '\n')
        test_loss_history = np.append(test_loss_history, np.mean(test_loss))
        test_accuracy_history = np.append(test_accuracy_history, np.mean(test_accuracy))
        # save result

        with open(output_file, 'a') as ref:
            print('test loss:', np.mean(test_loss), file=ref)
            print('test accuracy:', np.mean(test_accuracy), file=ref)
with open(output_file, 'a') as ref:
    print("Cover "+ str(len(selected_idx)) + " subjects", file=ref)
    print("Average accuracy:", np.mean(test_accuracy_history), file=ref)