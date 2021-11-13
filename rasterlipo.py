# functional oulipo


from rnntext import RNNText, OneStep

import argparse
import json
import itertools
import os
import math
import tensorflow as tf



class RasterLipo(RNNText):

  def raster_sample(self, initial, warmup, outfile, temperature, linelength, lines, lipofn):
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)

    self.model.load_weights(latest)

    one_step_model = OneStep(self.model, self.chars_from_ids, self.ids_from_chars, temperature)

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
    return 0
  else:
    return 1


def liposets(colours):
  lipo = {}
  lipo['red'] =  colours['yellow'] + colours['green'] + colours['blue']
  lipo['orange'] = colours['green'] + colours['blue'] + colours['purple']
  lipo['yellow'] = colours['blue'] + colours['purple'] + colours['red']
  lipo['green'] = colours['red'] + colours['orange'] + colours['purple']
  lipo['blue'] = colours['red'] + colours['orange'] + colours['yellow']
  lipo['purple'] = colours['orange'] + colours['yellow'] + colours['green'] 
#  lipo['black'] = colours['red'] + colours['orange'] + colours['yellow'] + colours['green'] + colours['blue'] + colours['purple']
  return lipo


def load_config(colourfile):
  with open(colourfile, 'r') as cfh:
    cf = json.load(cfh)
  return cf

def lipo_set(glyphs):
  s = glyphs.upper() + glyphs.lower()
  return list(set(s))


def interpolipo(a, b, k):
  """
Interpolation between two sets of forbidden characters a and b where k goes
from 0 (a) to 1 (b).

The returned value is a dictionary of weights suitable for remasking the one-step modeller
"""
  weights = {}
  common = set(a).intersection(set(b))
  for c in common:
    weights[c] = -float('inf')
  a0 = set(a) - common
  b0 = set(b) - common
  if k == 0:
    for g in a0:
      weights[g] = -float('inf')
    return weights
  if k == 1:
    for g in b0:
      weights[g] = -float('inf')
    return weights
  for g in a0:
    weights[g] = -math.tan(math.pi * 0.5 * (1 - k))
  for g in b0:
    weights[g] = -math.tan(math.pi * 0.5 * k)
  return weights


def interpolipo2(a, b, weight, k):
  """
Interpolation between two sets of promoted characters a and b where k goes
from 0 (a) to 1 (b).

The returned value is a dictionary of weights suitable for remasking the one-step modeller
"""
  weights = {}
  common = set(a).intersection(set(b))
  for c in common:
    weights[c] = weight
  a0 = set(a) - common
  b0 = set(b) - common
  if k <= 0:
    for g in a0:
      weights[g] = weight
    return weights
  if k >= 1:
    for g in b0:
      weights[g] = weight
    return weights
  for g in a0:
    weights[g] = weight * (1 - k)
  for g in b0:
    weights[g] = weight * k
  return weights


def make_gradient_fn(a, b, w):
  return lambda k: interpolipo2(a, b, w, k)

def make_circle_fn(width, height, gradientf):
  return lambda x, y: gradientf(math.dist([2 * (x - width * 0.5) / width, 2 * (y - height * 0.5) / height], [0,0]))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config", type=str, default="config.json", help="JSON config file")
  parser.add_argument("-n", "--name", type=str, required=True, help="name of this RNN")
  parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sample temperature")
  parser.add_argument("-o", "--outdir", type=str, default="output",  required=True, help="Output directory")
  args = parser.parse_args()
  cf = load_config(args.config)
  rnn = RasterLipo(args.name)
  rnn.init('')

  colours = cf['colours']
  width = int(cf['page']['width'])
  height = int(cf['page']['height'])

  os.makedirs(os.path.join('output', args.outdir))

  for c in itertools.combinations(colours.keys(), 2):
    outfile = os.path.join('output', args.outdir, f'{c[0]}-{c[1]}.txt')
    a = lipo_set(colours[c[0]])
    b = lipo_set(colours[c[1]])
    gfn = make_gradient_fn(b, a, 8)
    lf = make_circle_fn(width, height, gfn)
    #lf = lambda x, y: interpolipo2(a, b, 8, y / height)
    print(f'writing {outfile}: {c[1]} -> {c[0]}')
    rnn.raster_sample('test', 80, outfile, args.temperature, width, height, lf)