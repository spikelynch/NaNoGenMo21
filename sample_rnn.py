
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time
import argparse

# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
#from tensorflow.python.compiler.mlcompute import mlcompute

# Select CPU device.
#mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and 'any'.


#path_to_file = tf.keras.utils.get_file('kailung.txt', 'https://www.gutenberg.org/files/1076/1076-0.txt')


# TODO make these configurable

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
SEQ_LENGTH = 100
EPOCHS = 20
CHECKPOINTS_DIR = 'training_checkpoints'
DATA_DIR = 'data'

# keras.Model

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


# Sampler

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states






def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text


# TODO - save out the vocab stuff so that sampling doesn't have to regenerate it

class RNNText():
  """
A class which builds a model for an RNN to simulate text and either trains it
or samples it from a set of checkpoints.

"""

  def __init__(self, name):
    self.name = name
    self.checkpoint_dir = os.path.join('.', CHECKPOINTS_DIR, name)
    if not os.path.isdir(self.checkpoint_dir):
      print(f'Making checkpoint directory {self.checkpoint_dir}')
      os.makedirs(self.checkpoint_dir)
    self.data_file = os.path.join('.', DATA_DIR, name + '.txt')


  def text_from_ids(self, ids):
    return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)


  def init(self, training_data):
    if not os.path.isfile(self.data_file):
      tf.keras.utils.get_file(self.data_file, training_data)

    print("file = " + self.data_file)

    # Read, then decode for py2 compat.
    self.text = open(self.data_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    print(f'Length of text: {len(self.text)} characters')

    # The unique characters in the file
    self.vocab = sorted(set(self.text))
    print(f'{len(self.vocab)} unique characters')

    self.ids_from_chars = preprocessing.StringLookup(vocabulary=list(self.vocab), mask_token=None)
    self.chars_from_ids = preprocessing.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    self.model = MyModel(
      # Be sure the vocabulary size matches the `StringLookup` layers.
      vocab_size=len(self.ids_from_chars.get_vocabulary()),
      embedding_dim=EMBEDDING_DIM,
      rnn_units=RNN_UNITS)



  def train(self, epochs):
    self.all_ids = self.ids_from_chars(tf.strings.unicode_split(self.text, 'UTF-8'))
    self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.all_ids)
    self.examples_per_epoch = len(self.text)//(SEQ_LENGTH + 1)
    self.sequences = self.ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    self.dataset = self.sequences.map(split_input_target)

    for input_example, target_example in self.dataset.take(1):
      print("Input :", self.text_from_ids(input_example).numpy())
      print("Target:", self.text_from_ids(target_example).numpy())

    self.dataset = (self.dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocabulary in chars
    self.vocab_size = len(self.vocab)

    for input_example_batch, target_example_batch in self.dataset.take(1):
      example_batch_predictions = self.model(input_example_batch)
      print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input:\n", self.text_from_ids(input_example_batch[0]).numpy())
    print()
    print("Next Char Predictions:\n", self.text_from_ids(sampled_indices).numpy())

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    mean_loss = example_batch_loss.numpy().mean()
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", mean_loss)

    print(tf.exp(mean_loss).numpy())

    self.model.compile(optimizer='adam', loss=loss)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "chkpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

    # print(dataset)

    self.history = self.model.fit(self.dataset, epochs=epochs, callbacks=[checkpoint_callback])


  def sample(self, initial, temperature, length):
    print(f'loading latest checkpoint from {self.checkpoint_dir}')
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)
    print(f'latest checkpoint = {latest}')
    assert(latest)

    self.model.load_weights(latest)

    one_step_model = OneStep(self.model, self.chars_from_ids, self.ids_from_chars, temperature)

    start = time.time()
    states = None
    next_char = tf.constant([initial])
    result = [next_char]

    for n in range(length):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name", type=str, required=True, help="name of this RNN")
  parser.add_argument("-u", "--url", type=str, help="url of text to train from")
  parser.add_argument("-f", "--fit", action='store_true', help="Train the model")
  parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="Number of training epochs to train for")
  parser.add_argument("-s", "--sample", action='store_true', help="sample from latest checkpoint")
  parser.add_argument("-i", "--initial", type=str, default="start", help="Initial string")
  parser.add_argument("-l", "--length", type=int, default=1000, help="Length of sample in characters")
  parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sample temperature")
  args = parser.parse_args()
  print(args)
  rnn = RNNText(args.name)
  rnn.init(args.url)
  if args.fit:
    print("Training")
    rnn.train(args.epochs)
  if args.sample:
    print("Sampling")
    rnn.sample(args.initial, args.temperature, args.length)