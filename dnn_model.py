# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

class DNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, vocab_size, embed_size, hidden_size, is_training, tag_flag=True, initializer=initializers.xavier_initializer(), clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        
        # define placeholder
        self.query= tf.placeholder(tf.int32, [None, None], name="query")
        self.query_mask = tf.cast(tf.not_equal(self.query, 0), tf.float32)
        self.cid= tf.placeholder(tf.int32, [None, None], name="cid")
        self.cid_mask = tf.cast(tf.not_equal(self.cid, 0), tf.float32)
        self.product= tf.placeholder(tf.int32, [None, None], name="product")
        self.product_mask = tf.cast(tf.not_equal(self.product, 0), tf.float32)
        self.cid_neg= tf.placeholder(tf.int32, [None, None], name="cid_neg")
        self.cid_neg_mask = tf.cast(tf.not_equal(self.cid_neg, 0), tf.float32)
        self.product_neg= tf.placeholder(tf.int32, [None, None], name="product_neg")
        self.product_neg_mask = tf.cast(tf.not_equal(self.product_neg, 0), tf.float32)
        #self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        #self.length = tf.cast(self.length, tf.int32)
        

        self.intention_label = tf.placeholder(tf.int32, [None,1], name="intention_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        
        # init model 
        self.instantiate_dnn()
        self.logits = self.inference_dnn()
        
        # compute acc 
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(0.,self.logits), tf.float32), name="Accuracy")
            
        # compute loss
        self.loss_val= self.loss()

        if not is_training:
            return
        self.train_op = self.train_nodecay()
       
    def share_dnn(self, embedds, mask ):
        terms_mask = tf.expand_dims(mask, -1)
        input_feature = tf.multiply(terms_mask, embedds)
        sentence_feature = tf.reduce_sum(input_feature, axis=1)
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(sentence_feature, 512, activation=tf.nn.tanh)
            h2 = tf.layers.dense(h1, 256, activation=tf.nn.tanh)
        #h_drop = tf.nn.dropout(h2, keep_prob=self.dropout_keep_prob)
        return h2   
    
         
    def augment(self, embed):
        with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(embed, self.hidden_size, activation=tf.nn.tanh)
        return h
 
    def inference_dnn(self):

        # get emb
        self.embedded_query = tf.nn.embedding_lookup(self.Embedding,self.query)
        self.embedded_cid = tf.nn.embedding_lookup(self.Embedding,self.cid)
        self.embedded_cid_neg = tf.nn.embedding_lookup(self.Embedding,self.cid_neg)

        self.embedded_product = tf.nn.embedding_lookup(self.Embedding,self.product)
        self.embedded_product_neg = tf.nn.embedding_lookup(self.Embedding,self.product_neg)
    
        # get query emb
        self.query_f = self.share_dnn(self.embedded_query, self.query_mask)

        # get cid emb
        self.cid_f = self.share_dnn(self.embedded_cid, self.cid_mask)
        self.cid_n_f = self.share_dnn(self.embedded_cid_neg, self.cid_neg_mask)
        

        # get product emb
        self.product_f = self.share_dnn(self.embedded_product, self.product_mask)
        self.product_n_f = self.share_dnn(self.embedded_product_neg, self.product_neg_mask)
        # normalize 
        self.query_f = tf.nn.l2_normalize(self.query_f, dim=1)
        self.cid_f = tf.nn.l2_normalize(self.cid_f, dim=1)
        self.cid_n_f = tf.nn.l2_normalize(self.cid_n_f, dim=1)

        self.product_f = tf.nn.l2_normalize(self.product_f, dim=1)
        self.product_n_f = tf.nn.l2_normalize(self.product_n_f, dim=1)


        # comput cos similar
        query_cid = tf.multiply(self.query_f,self.cid_f)
        self.query_cid = tf.reduce_sum(query_cid,axis=1)

        query_n_cid = tf.multiply(self.query_f,self.cid_n_f)
        self.query_n_cid = tf.reduce_sum(query_n_cid,axis=1)
        

        
        query_product = tf.multiply(self.query_f,self.product_f)
        self.query_product = tf.reduce_sum(query_product,axis=1)
        query_n_product = tf.multiply(self.query_f,self.product_n_f)
        self.query_n_product = tf.reduce_sum(query_n_product,axis=1)
        
        # margin: 0.7   triple loss
        cid_loss=self.query_n_cid - self.query_cid + 0.5 
        product_loss=self.query_n_product - self.query_product + 0.5 
        
        # focal loss
        cid_p_weight = tf.pow(1-self.query_cid, 2)
        cid_n_weight = tf.pow(tf.maximum(self.query_n_cid,0), 2)
 
 
        product_p_weight = tf.pow(1-self.query_product, 2)
        product_n_weight = tf.pow(tf.maximum(self.query_n_product,0), 2)
 
        cid_weight = cid_p_weight + cid_n_weight
        product_weight = product_p_weight + product_n_weight 
        
        with tf.name_scope("output"):
            cid_loss = tf.maximum(cid_loss, tf.zeros_like(cid_loss))
            product_loss = tf.maximum(product_loss, tf.zeros_like(product_loss))
           
            # weighted  
            logits = cid_weight*cid_loss + product_weight*product_loss
            return logits
            

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(self.logits)
            l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if 'bias' not in v.name], name="l2_loss") * l2_lambda
           sum_loss = loss + l2_loss 
        return sum_loss


    def train_nodecay(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        train_op = opt.minimize(self.loss_val, self.global_step)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    
    def instantiate_dnn(self):
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)


