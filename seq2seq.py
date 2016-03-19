"""Code for interfacing with model.

Some code taken from TensorFlow Tutorials
https://www.tensorflow.org/versions/r0.7/tutorials/seq2seq/index.html
"""

from __future__ import print_function

import data
import seq2seq_model

import numpy as np
import tensorflow as tf
import math
import pickle
import random
import sys
import time
import os

OUT_DIR = 'out'
OUT_PREFIX = 'grammar.ckpt'
STEPS_PER_CHECKPOINT = 1

NHIDDEN = 32
NLAYERS = 1
MAX_GRAD = 5.0
BATCH_SIZE = 32
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.99


def bucket_sizes(sequences, delta=5):
    """Returns buckets and how many pairs of sequences end up in each."""
    buckets = {}
    for s1, s2 in sequences:
        bucket_id = (delta * ((1 + len(s1)) // delta),
                     delta * ((1 + len(s2)) // delta))
        if bucket_id not in buckets:
            buckets[bucket_id] = 0
        buckets[bucket_id] += 1

    return buckets


def bucketize(buckets, data_set):
    """Returns data_set sorted into buckets based on lengths of sequences."""
    ret = [[] for _ in buckets]
    for seq1, seq2 in data_set:
        len1 = len(seq1) + 1  # add one for EOS
        len2 = len(seq2) + 1  # add one for GO
        for bid, (bsize1, bsize2) in enumerate(buckets):
            if len1 <= bsize1 and len2 <= bsize2:
                padding1 = bsize1 - len1
                padding2 = bsize2 - len2
                ret[bid].append(
                    (seq1 + [data.EOS_ID] + [data.PAD_ID] * padding1,
                     seq2 + [data.EOS_ID] + [data.PAD_ID] * padding2))
                break

    return ret


def create_model(session, load_checkpoint, *args, **kwargs):
    """Create model."""
    model = seq2seq_model.Seq2SeqModel(*args, **kwargs)
    model.batch_size = 1
    if load_checkpoint:
        ckpt = tf.train.get_checkpoint_state(OUT_DIR)
        assert ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path), 'bad checkpoint file'
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.initialize_all_variables())
    return model

default_buckets = [(10, 30), (20, 90), (25, 120), (40, 170)]

def train(data_file, buckets=default_buckets):
    """Create and train a model."""
    with open(data_file, 'rb') as f:
        loaded_data = pickle.load(f)
    train_data, test_data, words, word2idx, grams, gram2idx = \
        tuple(loaded_data[k] for k in
              ['train_data', 'test_data', 'words', 'word2idx',
               'grams', 'gram2idx'])

    train_set = bucketize(buckets, train_data)
    test_set = bucketize(buckets, test_data)

    enc_vocab_size = len(words)
    dec_vocab_size = len(grams)

    with tf.Session() as sess:
        print('creating model')
        model = create_model(sess, False,
            enc_vocab_size, dec_vocab_size, buckets,
            NHIDDEN, NLAYERS, MAX_GRAD, BATCH_SIZE,
            LEARNING_RATE, LEARNING_RATE_DECAY,
            forward_only=False)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
        print('bucket sizes (train):', train_bucket_sizes)
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            rand = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > rand])
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
            loss += step_loss / STEPS_PER_CHECKPOINT
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % STEPS_PER_CHECKPOINT == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(OUT_DIR, OUT_PREFIX)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()


def decode(data_file, buckets=default_buckets):
    """Create and train a model."""
    with open(data_file, 'rb') as f:
        loaded_data = pickle.load(f)
    train_data, test_data, words, word2idx, grams, gram2idx = \
        tuple(loaded_data[k] for k in
              ['train_data', 'test_data', 'words', 'word2idx',
               'grams', 'gram2idx'])

    enc_vocab_size = len(words)
    dec_vocab_size = len(grams)

    with tf.Session() as sess:
        print('creating model')
        model = create_model(sess, True,
            enc_vocab_size, dec_vocab_size, buckets,
            NHIDDEN, NLAYERS, MAX_GRAD, BATCH_SIZE,
            LEARNING_RATE, LEARNING_RATE_DECAY,
            forward_only=True)

        # decode from standard input
        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            tokens = data.tokenize_sentence(sentence, lower=True)
            token_ids = [word2idx.get(tok, data.UNK_ID) for tok in tokens]

            bucket_id = min([b for b in xrange(len(buckets))
                             if buckets[b][0] >= len(token_ids) + 1])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            print(output_logits[0].shape)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data.EOS_ID)]

            # Print out grammar sentence corresponding to outputs.
            print(' '.join([grams[output] for output in outputs]))
            print('> ', end='')
            sys.stdout.flush()
            sentence = sys.stdin.readline()
