# functional oulipo


from rnntext import RNNText

import argparse


class RasterLipo(RNNText):

  def raster_sample(self, initial, warmup, outfile, temperature, linelength, lines, lipofn):
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)

    self.model.load_weights(latest)

    one_step_model = OneStep(self.model, self.chars_from_ids, self.ids_from_chars, temperature)

    start = time.time()
    states = None
    next_char = tf.constant([initial])
    output = []

    for n in range(warmup):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)

    for y in range(lines):
      x = 0
      result = ''
      lastspace = False
      while x < linelength:
        one_step_model.mask(lipofn(x, y))
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        schar = next_char[0].numpy().decode('utf-8')
        if schar.isspace():
          if not lastspace:
            result += ' '
            x += 1
            lastspace = True
        else:
          result += schar
          x += 1
          lastspace = False
      print(result)
      output.append(result)

    with open(outfile, 'w') as of:
      for l in output:
        of.write(l + "\n")


def checker(x, y):
  u = x // 20
  v = y // 10
  if (u + v) % 2 == 0:
    return "eiouEIOU"
  else:
    return "aiouAIOU"


def checker2(x, y):
  u = x // 20
  v = y // 20
  if (u + v) % 2 == 0:
    return "acemnorsuvwxzABCDEFGHIJKLMNOPQRSTUVXWYZ"
  else:
    return "bdfghijklpqty"


def nullipo(x, y):
  return "Ee"


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name", type=str, required=True, help="name of this RNN")
  parser.add_argument("-i", "--initial", type=str, default="start", help="Initial string")
  parser.add_argument("-l", "--length", type=int, default=1000, help="Length of sample in characters")
  parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sample temperature")
  args = parser.parse_args()
  rnn = RasterLipo(args.name)
  rnn.init('')
  rnn.sample(args.initial, args.temperature, args.length, "")