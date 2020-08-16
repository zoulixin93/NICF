#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
from .base import basic_model
import tensorflow as tf
from .base import *

class FA(basic_model):
    def _create_placeholders(self):
        self.utype = tf.placeholder(tf.int32, (None,), name='uid')
        self.p_rec = [tf.placeholder(tf.int32,(None,None,), name="p"+str(i)+"_rec") for i in range(6)]
        self.pt = [tf.placeholder(tf.int32,(None,2),"p"+str(i)+"t") for i in range(6)]
        self.rec = tf.placeholder(tf.int32, (None,), name='iid')
        self.target = tf.placeholder(tf.float32,(None,),name="target")

    def _update_placehoders(self):
        self.placeholders["all"] = {"uid":self.utype,
                                    "iid":self.rec,
                                    "goal":self.target}
        for i in range(6):
            self.placeholders["all"]["p"+str(i)+"_rec"] = self.p_rec[i]
            self.placeholders["all"]["p"+str(i)+"t"] = self.pt[i]
        self.placeholders["predict"] = {item: self.placeholders["all"][item] for item in ["uid"] + ["p"+str(i)+"_rec" for i in range(6)] + ["p"+str(i)+"t" for i in range(6)]}
        self.placeholders["optimize"] = self.placeholders["all"]

    def _create_inference(self):
        p_f = [tf.Variable(np.random.uniform(-0.01, 0.01,(self.args.item_num,self.args.latent_factor)),
                                   dtype=tf.float32, trainable=True, name='item'+str(i)+'_feature') for i in range(6)]
        u_f = tf.Variable(np.random.uniform(-0.01, 0.01,(self.args.utype_num,self.args.latent_factor)),
                                   dtype=tf.float32, trainable=True, name='user_feature')
        u_emb = tf.nn.embedding_lookup(u_f, self.utype)
        self.p_rec = [tf.transpose(item,[1,0]) for item in self.p_rec]
        i_p_mask = [tf.expand_dims(tf.to_float(tf.not_equal(item, 0)), -1) for item in self.p_rec]


        self.p_seq = [tf.nn.embedding_lookup(p_f[i],self.p_rec[i]) for i in range(6)]
        for iii,item in enumerate(self.p_seq):
            for i in range(self.args.num_blocks):
                with tf.variable_scope("rate_"+str(iii)+"_num_blocks_"+str(i)):
                    item = multihead_attention(queries=normalize(item),
                                                   keys=item,
                                                   num_units=self.args.latent_factor,
                                                   num_heads=self.args.num_heads,
                                                   dropout_rate=self.args.dropout_rate,
                                                   is_training=True,
                                                   causality=True,
                                                   scope="self_attention_pos_"+str(i))

                    item = feedforward(normalize(item), num_units=[self.args.latent_factor, self.args.latent_factor],
                                           dropout_rate=self.args.dropout_rate, is_training=True,scope="feed_forward_pos_"+str(i))
                    item *= i_p_mask[iii]
        self.p_seq = [normalize(item) for item in self.p_seq]

        p_out = [tf.gather_nd(tf.transpose(self.p_seq[i],[1,0,2]), self.pt[i]) for i in range(6)]
        context = tf.concat(p_out,1)
        hidden = tf.layers.dense(context,self.args.latent_factor,activation=tf.nn.relu)
        self.pi = tf.layers.dense(hidden, self.args.item_num, trainable=True)

    def _build_actor(self,context,name,trainable):
        with tf.variable_scope(name):
            a_prob = tf.layers.dense(context, self.args.item_num, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def _create_optimizer(self):
        a_indices = tf.stack([tf.range(tf.shape(self.rec)[0], dtype=tf.int32), self.rec], axis=1)
        self.npi = tf.gather_nd(params=self.pi, indices=a_indices)
        self.loss = tf.losses.mean_squared_error(self.npi,self.target)
        self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss)

    def optimize_model(self,sess,data):
        feed_dicts = self._get_feed_dict("optimize",data)
        return sess.run([self.loss,self.npi,self.optimizer],feed_dicts)[:2]

    def predict(self,sess,data):
        feed_dicts = self._get_feed_dict("predict", data)
        return sess.run(self.pi, feed_dicts)



