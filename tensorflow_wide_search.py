"""
TensorFlow 1 example.

This is an excerpt from the source code for my master's thesis, 'Visual blending in
sketch drawings using variational autoencoders'. Thesis text (in Spanish) is available
online through the library of the National Autonomous University of Mexico at:
http://tesis.unam.mx.

This model searches, using stochastic gradient descent, the latent space of a pre-trained
variational auto-encoder (VAE) for a sub-space containing a higher concentration of
latent vectors that have certain caracteristics as defined by a separate classifier model.

Specifically, it searches for latent vectors which translate to sketch drawings which
humans interpret as visual blends between two categories of sketch drawing. Ex. houses
and washing machines or ducks and rollerskates.

Abreviations:
    cls Classifier.
    kl  Kullback-Leibler divergence.
    lr  Learning rate.
    srm sketch-rnn model, a variational auto-encoder that reconstructs sketch
            drawings made by humans which was introduced by David Ha and Douglas
            Eck in 2017.
    v   Latent vector before modification.
    z   Latent vector after modification.

Copyright 2018 Derek Cheung. All Rights Reserved.
"""

import os
import sys
import time
import random

import tensorflow as tf
import numpy as np
import sketch_rnn_train

import rnn
import utils
import tbatcher

# these are references to other files in my thesis project
import model as sketch_rnn
from trainer import Trainer
import classifier


def get_default_hparams():
    # TensorFlow hyperparameters helper
    hparams = tf.contrib.training.HParams(
        is_training=False,
        # directories
        save_dir="logs",
        srm_name="house-and-washing-machine",
        cls_name="classifier-house-and-washing-machine",
        test_name="search-wide-house-and-washing-machine",
        # params that will be auto-filled. we deiifne them
        # here that they are automatically saved and restored
        data_set=["auto"],
        labels=["auto"],
        label_width="auto",
        comparison_classes=["auto"],
        z_size="auto",
        # configuration params
        batch_size=100,
        log_loss_weight=0.1,
        kl_weight=0.1,
        kl_tolerance=0.2,
        learning_rate=1e-3,
        min_learning_rate=1e-5,
        decay_rate=0.9999,
        log_every_secs=10,
        save_every_secs=30,
        hidden_size=128,
        max_num_steps=100000,
    )
    return hparams


class SearchWide(Trainer):
    """
    This model takes no inputs. Instead, it generates a minibatch of latent
    vectors by sampling from a Gaussian distribution (as in the approach for a GAN),
    then passes these vectors through a network of fully connected neurons to
    parametrize a second Gaussian. Vectors are sampled from this second Gaussian,
    then by the classifier, which finally generates a loss term. The loss term
    can be backpropagated using the re-parametrization trick (as in the approach
    for a VAE).

    batch_v -> (mean,sigma) -> batch_z -> loss
    """

    # for easy access
    hps = None
    srm = None
    srm_s = None
    clsm = None
    dataset = None

    # output configuration
    loss_names = ["loss_gen", "log_loss", "kl_cost", "lr"]

    def __init__(self, hps):
        self.hps = hps

        # autofill some parameters
        if "auto" in hps.comparison_classes:
            hps.comparison_classes = hps_srm.data_set
        hps.z_size = hps_srm.z_size
        self.hps.label_width = self.clsm.hps.label_width
        self.hps.labels = self.clsm.hps.labels
        self.hps.data_set = self.clsm.hps.data_set

        # load in the sketch-rnn model
        self.srm_s = sketch_rnn.Model(hps_srm)  # load the sketch-rnn model

        # begin defining our model
        self.build()

    def build():
        with tf.variable_scope("searchwide"):
            # global step counter
            self.global_step = tf.get_variable(
                "global_step", dtype=tf.int32, initializer=0, trainable=False
            )

            # initial batch
            self.batch_v = tf.random_normal([self.hps.batch_size, self.hps.z_size])

            # just two hidden layers
            hidden = leaky_relu(tf.layers.dense(self.batch_v, self.hps.hidden_size))
            hidden = leaky_relu(tf.layers.dense(hidden, self.hps.hidden_size))

            # generate mean and -sigma
            self.mean = tf.layers.dense(hidden, self.hps.z_size)
            self.presig = tf.layers.dense(hidden, self.hps.z_size)
            self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.

            eps = tf.random_normal(
                (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32
            )
            # use mean and sigma to parametrize a gaussian distribution and
            # generate a batch of latent vectors
            self.batch_z = self.mean + tf.multiply(self.sigma, eps)

            # calculate the log loss for the original vectors with respect to the new
            # distribution. this keeps the new vectors relatively close to their
            # original values
            normal = tf.contrib.distributions.Normal(loc=self.mean, scale=self.sigma)
            self.log_loss = -tf.reduce_mean(normal.log_prob(self.batch_v))

            # this Kullback-Leibler divergence loss prefers the new mean and sigma
            # to be near 0 and 1, respectively
            kl_cost = -0.5 * tf.reduce_mean(
                (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)), 1
            )
            kl_cost = tf.maximum(kl_cost, self.hps.kl_tolerance)
            self.kl_cost = tf.reduce_mean(kl_cost)

        # load the external classifier out of the scope of this model,
        # feeding it our batch of latent vectors batch_z
        self.clsm = classifier.ZeroCritic(hps_cls, input_k=self.batch_z)

        # the classifier will populate self.clsm.logits

        # sigmoid cross entropy loss between the two vectors
        loss_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=self.clsm.logits
        )
        loss_gen = tf.reduce_mean(loss_gen)

        # composite loss term
        self.loss_gen = (
            loss_gen
            + (self.log_loss * self.hps.log_loss_weight)
            + (self.kl_cost * self.hps.kl_weight)
        )

        if self.hps.is_training:
            # learning rate starts big and gradually decreases
            self.lr = (hps.learning_rate - hps.min_learning_rate) * (
                hps.decay_rate ** tf.cast(self.global_step, tf.float32)
            ) + hps.min_learning_rate

            # adam optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=pelf.lr)
            keys = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="searchwide"
            )
            grads = optimizer.compute_gradients(loss, keys)
            self.train_op = optimizer.apply_gradients(
                grads, global_step=self.global_step, name=name
            )

    def restore_all(self, sess, best=False):
        """
        Restores a saved model
        """
        # this model
        saver = self.restore_self(sess, best, "searchwide")

        # classifier model
        keys = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="classifier")
        saver_cls = tf.train.Saver(keys)
        (save_path, save_path_best) = self.clsm.get_save_paths()
        saver_cls.restore(sess, save_path)
        global_step = sess.run(self.clsm.global_step)
        assert global_step != 0
        print("Restored classifier at global_step=%i" % global_step)

        # sketch-rnn model
        keys = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vector_rnn")
        saver_srm = tf.train.Saver(keys)
        try:
            saver_srm.restore(
                sess, os.path.join(self.hps.save_dir, self.hps.srm_name, "vector_best")
            )
        except:
            saver_srm.restore(
                sess, os.path.join(self.hps.save_dir, self.hps.srm_name, "vector")
            )
        global_step = sess.run(self.srm_s.global_step)
        assert global_step != 0
        print("Restored sketch-rnn model at global_step=%i" % global_step)

        return saver

    def train_step(self, sess, bw, step):
        # training step
        (loss_gen, loss_log, kl_cost, step, _) = sess.run(
            [
                self.loss_gen,
                self.log_loss,
                self.kl_cost,
                self.global_step,
                self.train_op_gen,
            ]
        )
        # log out these values
        return ([loss_gen, loss_log, kl_cost], step)


if __name__ == "__main__":
    raise RuntimeError(
        (
            "This is just a code excerpt and will not run without supporting"
            "files from the rest of the project."
        )
    )
    # load default parameters
    hps = get_default_hparams()
    hps_srm = get_srm_hparams(hps)
    hps_cls = get_cls_hparams(hps)

    model = ZeroFinder(hps, hps_srm, hps_cls)
    if hps.is_training:
        # the Trainer class will first call restore_all() before calling
        # train_step() iteratively until max_num_steps is reached
        model.train(save=True)
    else:
        # alternatively, the Trainer can call restore_all() then call other
        # code to sample from the trained search model
        model.sample(best=False)
