import tensorflow as tf
import load_cifar as cifar
import graph

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' ' # TODO: find better work around

#batch_size = 3
batch_size = 128
#epochs = 200
epochs = 7

logs_path = '/tmp/tensorflow_logs/test9'

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch_train = int(len(cifar.images_train) / batch_size)
    total_batch_val = int(len(cifar.images_val) / batch_size)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(epochs):
        learning_rate = graph.lr_def(epoch)
        #total_batch = 20 ######

        #training
        avg_loss_train = 0
        avg_acc_train = 0
        for i in range(total_batch_train):
            #offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
            offset = i * batch_size
            batch_data = cifar.images_train[offset:(offset + batch_size), :, :, :]
            batch_labels = cifar.labels_train[offset:(offset + batch_size), :]
            feed_dict_train = {graph.x: batch_data, graph.y: batch_labels, graph.is_training: True}
            train_acc, _, loss_train, summary = sess.run([graph.accuracy, graph.optimiser, graph.total_loss, graph.merged_summary_op], feed_dict=feed_dict_train)
            summary_writer.add_summary(summary, epoch * total_batch_train + i)
            avg_loss_train += loss_train / total_batch_train
            avg_acc_train += train_acc / total_batch_train
            #print ("Batch: {}, train_loss={}, train_accuracy={}".format(i+1, loss_train, train_acc))
        print("Epoch", (epoch + 1),":", "train loss =", "{:.3f}".format(avg_loss_train), "train accuracy =", "{:.3f}".format(avg_acc_train))

        # validation
        avg_loss_val = 0
        avg_acc_val = 0
        for i in range(total_batch_val):
            offset = i * batch_size
            batch_data_val = cifar.images_val[offset:(offset + batch_size), :, :, :]
            batch_labels_val = cifar.labels_val[offset:(offset + batch_size), :]
            feed_dict_val = {graph.x: batch_data_val, graph.y: batch_labels_val, graph.is_training: False}
            val_acc, loss_val = sess.run([graph.accuracy, graph.total_loss], feed_dict=feed_dict_val) #for validation loss is considered without the weight decay
            avg_loss_val += loss_val / total_batch_val
            avg_acc_val += val_acc / total_batch_val
        print("Epoch", (epoch + 1),":", "validation loss: {:.3f}".format(avg_loss_val), "validation accuracy: {:.3f}".format(avg_acc_val))

    #print("Epoch:", (epoch + 1), "train loss =", "{:.3f}".format(avg_loss), "train accuracy =",
          #"{:.3f}".format(avg_acc),
          #"validation loss: {:.3f}".format(loss_val), "validation accuracy: {:.3f}".format(val_acc))
    print("Run the command line:\n" \
         "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
    print("\nTraining complete!")
    #print(sess.run(graph.accuracy, feed_dict={graph.x: val_data, graph.y: val_label}))
    #print(sess.run(graph.accuracy, feed_dict={graph.x: test_data, graph.y: test_label}))