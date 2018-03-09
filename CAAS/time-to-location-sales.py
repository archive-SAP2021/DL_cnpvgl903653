# pylint: disable=multiple-statements, invalid-name, line-too-long, no-member
"""
Time-Location implementation.

"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import r2_score

import matplotlib
import matplotlib.pyplot as plt

def selectDF(raw_df_, selected_date_):
    """
    select data frame based on selected_date.
    """
    filter_data=raw_df_[raw_df_.Date==selected_date_]
    return filter_data[['Store', 'Sales', 'Attitude', 'Longtitude']]

def get_batch():
    df = pd.read_csv('data/time-series-location-tf/time-location-sales.gz', compression='gzip', parse_dates=['Date'],low_memory=False)
    df = df[df['Open'] == 1]
    # df.Store.replace(0, 1e-1, inplace=True)
    df.Sales.replace(0, 1e-2, inplace=True)
    df.sort_values(by='Date',inplace=True)
    df = df[['Date','Store', 'Sales', 'Attitude','Longtitude']]
    data = df.groupby(['Date'])
    all_items = [df for date,df in data]
    for i in range(0,len(all_items)-1):
        first,second = all_items[i],all_items[i+1]
        store1,store2 = first['Store'], second['Store']
        common = store1[store1.isin(store2)]
        loc = first[store1.isin(common)][['Attitude', 'Longtitude']].values.reshape(-1, 2)
        input = first[store1.isin(common)]['Sales'].values.reshape(-1,1)
        output = second[store2.isin(common)]['Sales'].values.reshape(-1, 1)
        yield loc,input,output

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def run_rnn_simple():
    ## modelling for tensorflow
    n_input=1
    n_output=1
    n_input_loc=2
    n_hidden1=200
    n_hidden2=200
    n_hidden3=200
    learning_rate = 1e-3
    L=tf.placeholder(tf.float32, [None, n_input_loc])
    S0_X=tf.placeholder(tf.float32, [None, n_input])
    S1_X=tf.placeholder(tf.float32, [None, n_output])
    # https://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html
    with tf.variable_scope('dnn', initializer=tf.contrib.layers.variance_scaling_initializer()):
        H1=fully_connected(L, n_hidden1, scope="hidden_layer_1")
        h1_output_size=int(H1.get_shape()[1])
        with tf.name_scope('location_to_sale1'):
            W_location_to_sale1=tf.Variable(name="location_to_sale1", initial_value=tf.truncated_normal([h1_output_size, n_hidden3], stddev=0.01))
            variable_summaries(W_location_to_sale1)
        with tf.name_scope('sale0_to_sale1'):
            W_sale0_to_sale1=tf.Variable(name="sale0_to_sale1", initial_value=tf.truncated_normal([n_input, n_hidden3], stddev=0.01))
            variable_summaries(W_sale0_to_sale1)
        H3=tf.matmul(H1, W_location_to_sale1) + tf.matmul(S0_X, W_sale0_to_sale1)
        # outputs = fully_connected(H3, n_output, scope="output", activation_fn=tf.nn.relu)
        outputs = fully_connected(H3, n_output, scope="output", activation_fn=None)
        loss = tf.reduce_mean(tf.square(outputs - S1_X)) # MSE
        y2yhat_loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - S1_X))) # tf.sqrt(y^2-y-hat^2)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
        init=tf.global_variables_initializer()
        with tf.name_scope('performance'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('y2yhat_loss', y2yhat_loss)

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            init.run()
            epoch=0
            batch_size=10
            alldata=list(get_batch())
            data_len=len(alldata)
            for loc,x,y in alldata:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                for iteration in range(batch_size):
                    for _ in range(10000):
                        summary, _, mse, true_loss, y_pred=sess.run([merged, training_op, loss, y2yhat_loss, outputs], feed_dict={L: loc, S0_X: x, S1_X: y}, options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % (epoch*batch_size+iteration))
                    train_writer.add_summary(summary, (epoch*batch_size+iteration))
                print("\tMSE on Training Set on epoch {}: {}, r2_score: {}".format(epoch, mse, r2_score(y_pred, y)))
                if epoch == data_len-1:
                    # if the last epoch
                    x = range(0, len(y))
                    plt.plot(x, y.flatten(), "*", markersize=4, label="target")
                    plt.plot(x, y_pred.flatten(), 'r.', markersize=4, label="prediction")
                    plt.show()
                epoch+=1
            train_writer.close()

def main(_):
    # Prepare log dir from the beginning.
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_rnn_simple()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/rnn-ts/logs/time-to-location-sales'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    