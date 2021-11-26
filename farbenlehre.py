# functional oulipo


from rnntext import RNNText, OneStep

import argparse
import json
import itertools
import os
import math
import yaml
import tensorflow as tf




def test_fn(label, lipofn, width, height):
  for x in range(0, width, 10):
    for y in range(0, height, 10):
      val = lipofn(x, y)
      print(f'>> {label} {x} {y} {val}')


class RasterLipo(RNNText):

  def raster_sample(self, initial, warmup, outfile, temperature, linelength, lines, pagefns):
    latest = tf.train.latest_checkpoint(self.checkpoint_dir)

    self.model.load_weights(latest)

    one_step_model = OneStep(self.model, self.chars_from_ids, self.ids_from_chars, temperature)

    states = None
    next_char = tf.constant([initial])

    for n in range(warmup):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)

    for page in pagefns:
      label = page[0]
      lipofn = page[1]
      y = 0
      while y < lines:
        x = 0
        result = ''
        whitespace = ''
        line = True
        while line:
          one_step_model.mask(lipofn(x, y))
          next_char, states = one_step_model.generate_one_step(next_char, states=states)
          schar = next_char[0].numpy().decode('utf-8')
          if schar.isspace():
            if len(whitespace) > 0:
              if whitespace == '\r\n':
                result += "\n"
                line = False
                whitespace = ''
                y += 2
              else:
                whitespace += schar
            else:
              result += ' '
              whitespace += schar
              x += 1
              if x > linelength:
                line = False
                y += 1
          else:
            result += schar
            x += 1
            whitespace = ''
        print(result)
        outfile.write(result + "\n")
      print("```\n\n\\newpage\n```\n")
      outfile.write("```\n\n\\newpage\n```\n")






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

def make_circle_gradient_fn(width, height, gradientf):
  return lambda x, y: gradientf(math.dist([2 * (x - width * 0.5) / width, 2 * (y - height * 0.5) / height], [0,0]))


def circle_gradient(width, height, x, y, k):
  return k * (math.dist([2 * (x - width * 0.5) / width, 2 * (y - height * 0.5) / height], [0,0]) - 1) + 1


def int_circle(width, height, x, y):
  if math.dist([2 * (x - width * 0.5) / width, 2 * (y - height * 0.5) / height], [0,0]) < 0.8:
    return 0
  else:
    return 1




def make_checker_fn(xsize, ysize, gradientf):
  return lambda x, y: gradientf(checker(xsize, ysize, x, y))

def checker(xsize, ysize, x, y):
  u = x // xsize
  v = y // ysize
  if (u + v) % 2 == 0:
    return 0
  else:
    return 1




def party_per_bend(width, height, x, y):
  if x / width < y / height:
    return 1
  else:
    return 0

def party_per_bend_gradient(width, height, x, y, k):
  v =  k * (y / height - x / width) * 0.5 + 0.5
  return v



def make_constraint(pattern, width, height, gradientf):
  if pattern == 'gradient':
    return lambda x, y: gradientf(y / height)
  elif pattern == 'circle':
    return lambda x, y: gradientf(int_circle(width, height, x, y))
  elif pattern == 'party':
    return lambda x, y: gradientf(party_per_bend(width, height, x, y))

def make_k_constraint(pattern, width, height, k, gradientf):
  if pattern == 'circle':
    return lambda x, y: gradientf(circle_gradient(width, height, x, y, k))
  elif pattern == 'party':
    return lambda x, y: gradientf(party_per_bend_gradient(width, height, x, y, k))



def make_permutations(seq, remain):
  """
  Recursively generates a path through permutations of all pairs of colours such that
  the last of pair n and the first of pair n + 1 match
  """

  tail = remain[:]
  start = seq[-1]
  possibilities = [ p for p in tail if p[0] == start[1] ]
  if possibilities:
    p = possibilities[-1]
    r = [ q for q in tail if q[0] != p[0] or q[1] != p[1] ]
    if r:
      return make_permutations(seq + [p], r)
    else:
      return seq + [p]
  else:
    return False

def get_permutations(colours):
  c = list(itertools.permutations(colours, 2))
  return make_permutations([ c[0] ], c[1:])

def complement(colour, base, seq):
  s = [ c for c in seq if c != base ]
  l = len(s)
  return s[(s.index(colour) + l // 2) % l]






if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config", type=str, default="config.json", help="JSON config file")
  parser.add_argument("-n", "--name", type=str, required=True, help="name of this RNN")
  parser.add_argument("-o", "--outdir", type=str, default="output",  required=True, help="Output directory")
  args = parser.parse_args()
  cf = load_config(args.config)
  rnn = RasterLipo(args.name)
  rnn.init('')

  colours = cf['colours']
  corder = cf['order']
  k = cf['steepness']
  strength = int(cf['strength'])
  width = int(cf['page']['width'])
  height = int(cf['page']['height'])
  temperature = float(cf['temperature'])

  os.makedirs(os.path.join('output', args.outdir))

  outfile = os.path.join('output', args.outdir, cf['output'])

  sequence = get_permutations(corder)

  lipofns = []
  for pair in sequence:
    a = lipo_set(colours[pair[0]])
    b = lipo_set(colours[pair[1]])
    comp = complement(pair[0], pair[1], corder)
    c = lipo_set(colours[comp])
    gfn1 = make_gradient_fn(a, b, strength)
    lipofns.append((f'diag: {pair[0]}->{pair[1]}', make_k_constraint('party', width, height, k, gfn1)))
    gfn2 = make_gradient_fn(c, b, strength)
    lipofns.append((f'circle: {pair[1]}, {comp}', make_k_constraint('circle', width, height, k, gfn2)))

  with open(outfile, 'w') as of:
    of.write('---\n' + yaml.dump(cf['latex']) + '---\n\\newpage\n```\n')
    rnn.raster_sample("GOETHE'S THEORY OF COLOURS", 0, of, temperature, width, height, lipofns[:6])
    of.write('```\n\n')