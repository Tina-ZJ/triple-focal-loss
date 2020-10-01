import tensorflow as tf
import numpy as np
import fileinput
import os
import pickle as pick


def extract_parameter(modelPath):
    nw = open(os.path.join(modelPath, "weight.names"), 'w')
    out_f = open(os.path.join(modelPath, "model.w"), "w")
    saver = tf.train.import_meta_graph(modelPath)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device("/cpu:0"):
            saver.restore(sess, modelPath)
            for var in tf.trainable_variables():
                data = sess.run(var)
                if len(data.shape) == 1:
                    data = data.reshape(1, data.shape[0])
                name = str(var.name).split("/")[-1]
                nw.write(name+" " + str(data.shape) + "\n")
                if 'Embedding' in name:
                    data = data.T
                    np.savetxt(out_f, data, fmt='%s', newline='\n')
    out_f.close()
    nw.flush()
    nw.close()

if __name__ == "__main__":
    extract_parameter("./checkpoint/dnn/model.ckpt-8")
