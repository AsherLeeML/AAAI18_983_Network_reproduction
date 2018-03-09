from model_utils.cnn import cnn
import sklearn
from sklearn import metrics
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time

from model_utils import data_utils

# np.random.seed(33)

ignore_idx = 89

######################
#  model parameter
######################

kernel_1st = [3, 3, 3]
kernel_2nd = [3, 3, 3]
kernel_3rd = [3, 3, 3]

kernel_stride = 1
conv_channel_num = 32

pooling_para = ['None']*3

pooling_stride = 'None'

fc_size = 1024

##########################
# dataset parameters
##########################

input_channel_num = 1
window_size = 10
input_depth = window_size

input_height = 10
input_width = 11

result_path = './results/3dcnn/'

labels_NB = 4

dataset_dir = './rep983/processed/'

test_subject = 1


class CNN_3D():
    def __init__(self):
    ######################################
    # training parameters
    ######################################

        self.learning_rate = 1e-4
        self.training_epochs = 20
        self.batch_size = 512
        self.dropout_prob = 0.5
        self.enable_penalty = True
        self.lambda_loss_amount = 0.0005
        self.accuracy_batch_size = 200

        #################################################
        #### model build
        #################################################

        self.cnn_3d = cnn()
        self.X = tf.placeholder(tf.float32, shape=[None, input_depth, input_height, input_width, input_channel_num], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, labels_NB], name='Y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.conv_layer1 = self.cnn_3d.apply_conv3d(self.X,
                                          kernel_1st[0], kernel_1st[1], kernel_1st[2],
                                          input_channel_num, conv_channel_num, kernel_stride)
        #pool_layer1 = cnn_3d.apply_max_pooling3d(conv_layer1,
        #                                         pooling_para[0], pooling_para[1], pooling_para[2],
        #                                         pooling_stride)
        self.conv_layer2 = self.cnn_3d.apply_conv3d(self.conv_layer1,
                                          kernel_2nd[0], kernel_2nd[1], kernel_2nd[2],
                                          conv_channel_num, conv_channel_num*2, kernel_stride)
        #pool_layer2 = cnn_3d.apply_max_pooling3d(conv_layer2,
        #                                         pooling_para[0], pooling_para[1], pooling_para[2],
        #                                         pooling_stride)
        self.conv_layer3 = self.cnn_3d.apply_conv3d(self.conv_layer2,
                                          kernel_3rd[0], kernel_3rd[1], kernel_3rd[2],
                                          conv_channel_num*2, conv_channel_num*4, kernel_stride)
        #pool_layer3 = cnn_3d.apply_max_pooling3d(conv_layer3,
        #                                         pooling_para[0], pooling_para[1], pooling_para[2],
        #                                         pooling_stride)
        self.shape = self.conv_layer3.get_shape().as_list()
        self.conv_layer3_flat = tf.reshape(self.conv_layer3, [-1, self.shape[1]*self.shape[2]*self.shape[3]*self.shape[4]])
        self.fc = self.cnn_3d.apply_fully_connect(self.conv_layer3_flat, self.shape[1]*self.shape[2]*self.shape[3]*self.shape[4], fc_size)
        self.fc_drop = tf.nn.dropout(self.fc, self.keep_prob)
        self.y_ = self.cnn_3d.apply_readout(self.fc_drop, fc_size, labels_NB)
        self.y_posi = tf.nn.softmax(self.y_, name='y_posi')
        self.y_pred = tf.argmax(self.y_posi, 1, name='y_pred')
        self.l2 = self.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        if self.enable_penalty:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.Y)) + self.l2
        else:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.Y), name='loss')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.y_), 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

############################
# random test begins:
############################


    def train_model(self, training_epochs, batch_size, shuffle=True, finetune=False, finetune_epochs=None):
        batch_size = batch_size
        accuracy_batch_size = self.accuracy_batch_size
        training_epochs = training_epochs
        enable_penalty = self.enable_penalty
        optimizer = self.optimizer
        cost = self.cost

        selected_idx = np.array(range(1, 110))
        np.random.shuffle(selected_idx)
        selected_idx = selected_idx[:20]

        test_loss_history = np.empty(shape=[0], dtype=float)
        test_accuracy_history = np.empty(shape=[0], dtype=float)


        for test_subject in selected_idx:
            if test_subject == ignore_idx:
                continue

            if finetune:
                pretrain_x, pretrain_y, finetune_x, finetune_y, test_x, test_y = data_utils.leave_one_finetune_split(test_subject, split=True)
                train_x, train_y = pretrain_x, pretrain_y
                print("Working on subject ", format(test_subject, "03d"))
                print("Train data shape:", train_x.shape)
                print("Train label shape:", train_y.shape)
                print('Finetune data shape:', finetune_x.shape)
                print('Finetune label shape:', finetune_y.shape)
                print("Test data shape:", test_x.shape)
                print('Test label shape:', test_y.shape)
                finetune_x = finetune_x.reshape(len(finetune_x), window_size, 10, 11, 1)
                finetune_y = np.asarray(pd.get_dummies(finetune_y), dtype=np.int8)


            else:
                train_x, train_y, test_x, test_y = data_utils.leave_one_finetune_split(test_subject)


            idx = np.array(range(len(train_y)))
            train_x = train_x[idx]
            train_y = train_y[idx]

            train_x = train_x.reshape(len(train_x), window_size, 10, 11, 1)
            test_x = test_x.reshape(len(test_x), window_size, 10, 11, 1)

            train_y = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
            test_y = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

            batchNB_per_epoch = train_x.shape[0] // batch_size
            train_accuracy_batchNB = train_x.shape[0]//accuracy_batch_size
            test_accuracy_batchNB = test_x.shape[0]//accuracy_batch_size

            if enable_penalty:
                reg_method = 'dropout+l2'
            else:
                reg_method = 'dropout'


            #######################################
            # train and test and result save
            #######################################

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as session:
                session.run(tf.global_variables_initializer())

                early_stopping_count = 0

                train_accuracy_save = np.zeros(shape=[0], dtype=float)
                test_accuracy_save  = np.zeros(shape=[0], dtype=float)
                train_loss_save     = np.zeros(shape=[0], dtype=float)
                test_loss_save      = np.zeros(shape=[0], dtype=float)

                for epoch in range(training_epochs):

                    cost_history = np.zeros(shape=[0], dtype=float)
                    for batch in range(batchNB_per_epoch):
                        offset = (batch * batch_size) % (train_y.shape[0] - batch_size)
                        batch_x = train_x[offset:(offset+batch_size), :, :, :, :]
                        batch_y = train_y[offset:(offset+batch_size), :]
                        _, c = session.run([optimizer, cost], feed_dict={self.X: batch_x, self.Y:batch_y, self.keep_prob: 1-self.dropout_prob})
                        cost_history = np.append(cost_history, c)

                    train_accuracy = np.zeros(shape=[0], dtype=float)
                    test_accuracy  = np.zeros(shape=[0], dtype=float)
                    train_loss     = np.zeros(shape=[0], dtype=float)
                    test_loss      = np.zeros(shape=[0], dtype=float)

                    for i in range(train_accuracy_batchNB):
                        offset = (i*accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size)
                        train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                        train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]
                        train_a, train_c = session.run([self.accuracy, cost], feed_dict={self.X: train_batch_x, self.Y: train_batch_y, self.keep_prob:1.0})
                        train_loss = np.append(train_loss, train_c)
                        train_accuracy = np.append(train_accuracy, train_a)
                    print('Epoch: ', epoch + 1, "\nTraining Cost:", np.mean(train_loss), "Training Accuracy:", np.mean(train_accuracy) )
                    train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
                    train_loss_save     = np.append(train_loss_save, np.mean(train_loss))

                    for i in range(test_accuracy_batchNB):
                        offset = (i*accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                        test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                        test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
                        test_a, test_c = session.run([self.accuracy, cost], feed_dict={self.X: test_batch_x, self.Y: test_batch_y, self.keep_prob: 1.0})
                        test_accuracy = np.append(test_accuracy, test_a)
                        test_loss = np.append(test_loss, test_c)
                    print('Test Cost:', np.mean(test_loss), "Test Accuracy:", np.mean(test_accuracy),'\n')
                    test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                    test_loss_save     = np.append(test_loss_save, np.mean(test_loss))

                    if epoch > 0:
                        if train_loss_save[epoch] < train_loss_save[epoch-1]:
                            if test_loss_save[epoch] > test_loss_save[epoch-1]:
                                early_stopping_count += 1
                            else:
                                early_stopping_count = 0
                        if early_stopping_count > 3:
                            print('Early stopping.')
                            break

                with open(result_path+'result.txt','a') as ref:
                    print('Subject id:', format(test_subject, '03d'), file=ref)
                    print('Final training loss:', np.mean(train_loss), "\tFinal Training Accuracy:", np.mean(train_accuracy), file=ref)

                ################# finetuning ##################
                if finetune:
                    print("---------------------------------------------Finetuning--------------------------------------------------:")
                    print('Samples used for finetuning:', len(finetune_y))
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
                            _, c = session.run([optimizer, cost], feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                             self.keep_prob: 1 - self.dropout_prob})
                            cost_history = np.append(finetune_cost_history, c)
                        train_accuracy = np.zeros(shape=[0], dtype=float)
                        test_accuracy = np.zeros(shape=[0], dtype=float)
                        train_loss = np.zeros(shape=[0], dtype=float)
                        test_loss = np.zeros(shape=[0], dtype=float)

                        for i in range(finetune_batch_per_epoch):
                            offset = (i*accuracy_batch_size) % (finetune_y.shape[0] - accuracy_batch_size)
                            train_batch_x = finetune_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                            train_batch_x = train_batch_x.reshape(len(train_batch_x) * window_size, 10, 11, 1)
                            train_batch_y = finetune_y[offset:(offset + accuracy_batch_size), :]
                            train_a, train_c = session.run([self.accuracy, cost], feed_dict={self.X: train_batch_x, self.Y: train_batch_y, self.keep_prob:1.0})
                            train_loss = np.append(train_loss, train_c)
                            train_accuracy = np.append(train_accuracy, train_a)
                        print('Epoch: ', epoch + 1, "\nFinetuning Cost:", np.mean(train_loss), "Finetuning Accuracy:", np.mean(train_accuracy))

                        finetune_accuracy_save = np.append(finetune_accuracy_save, np.mean(train_accuracy))
                        finetine_loss_save     = np.append(finetine_loss_save, np.mean(train_loss))

                        for i in range(test_accuracy_batchNB):
                            offset = (i*accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                            test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                            test_batch_x = test_batch_x.reshape(len(test_batch_x) * window_size, 10, 11, 1)
                            test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
                            test_a, test_c = session.run([self.accuracy, cost], feed_dict={self.X: test_batch_x, self.Y: test_batch_y, self.keep_prob: 1.0})
                            test_accuracy = np.append(test_accuracy, test_a)
                            test_loss = np.append(test_loss, test_c)
                        print('Test Cost:', np.mean(test_loss), "Test Accuracy:", np.mean(test_accuracy),'\n')

                        test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                        test_loss_save     = np.append(test_loss_save, np.mean(test_loss))

                        if epoch > 0:
                            if train_loss_save[epoch] < train_loss_save[epoch - 1]:
                                if test_loss_save[epoch] > test_loss_save[epoch - 1]:
                                    early_stopping_count += 1
                                else:
                                    early_stopping_count = 0
                            if early_stopping_count > 3:
                                print('Early stopping.\nFinetuning stops.')
                                break

                with open(result_path+'result.txt','a') as ref:
                    print('Final fintuning loss:', np.mean(train_loss), "\tFinal fintuning Accuracy:", np.mean(train_accuracy), file=ref)
            ################################################
            ## save result and model after training
            ################################################


                test_accuracy = np.zeros(shape=[0], dtype=float)
                test_loss     = np.zeros(shape=[0], dtype=float)
                # test_pred     = np.zeros(shape=[0], dtype=float)
                # test_true     = np.zeros(shape=[0, labels_NB], dtype=float)
                # test_posi     = np.zeros(shape=[0, labels_NB], dtype=float)

                for k in range(test_accuracy_batchNB):
                    offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                    test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
                    test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
                    test_batch_x = test_batch_x.reshape(len(test_batch_x) * window_size, 10, 11, 1)

                    test_a, test_c, test_p, test_r = session.run([self.accuracy, cost, self.y_pred, self.y_posi],
                                                                 feed_dict={self.X: test_batch_x, self.Y: test_batch_y, self.keep_prob: 1.0})
                    test_t = test_batch_y

                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)
                    # test_pred = np.append(test_pred, test_p)
                    # test_true = np.vstack([test_true, test_t])
                    # test_posi = np.vstack([test_posi, test_r])

                #test_pred_one_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
                #test_true_list    = tf.argmax(test_true, 1).eval()

                #test_recall = metrics.recall_score(test_true, test_pred_one_hot, average=None)

                #test_precision = metrics.precision_score(test_true, test_pred_one_hot, average=None)

                #test_f1 = metrics.f1_score(test_true, test_pred_one_hot, average=None)

                #confusion_matrix = metrics.confusion_matrix(test_true_list, test_pred)

                print("("+time.asctime(time.localtime(time.time())) + ')\nFinal Test Cost:', np.mean(test_loss), 'Final Test Accuracy:', np.mean(test_accuracy), '\n')


                test_loss_history = np.append(test_loss_history, np.mean(test_loss))
                test_accuracy_history = np.append(test_accuracy_history, np.mean(test_accuracy))

                with open(result_path+'result.txt','a') as ref:
                    print('test loss:', np.mean(test_loss), file=ref)
                    print('test accuracy:', np.mean(test_accuracy), file=ref)

        with open(result_path+'result.txt', 'a') as ref:
            print("Cover "+ str(len(selected_idx)) + " subjects", file=ref)
            print("Average accuracy:", np.mean(test_accuracy_history), file=ref)



if __name__ == '__main__':
    cnn_3d = CNN_3D()
    # cnn_3d.train_model(
    #     batch_size=512,
    #     training_epochs=50,
    #     finetune=False,
    #     finetune_epochs=None
    # )
    cnn_3d.train_model(
        batch_size=1024,
        training_epochs=100,
        finetune=True,
        finetune_epochs=100)
