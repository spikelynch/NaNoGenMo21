#!/bin/env python

# exploring combinations


import itertools, random



def find_all_sequence(seq, remain):
	tail = remain[:]
	start = seq[-1]
	possibilities = [ p for p in tail if p[0] == start[1] ]
	if possibilities:
		for p in possibilities:
			r = [ q for q in tail if q[0] != p[0] or q[1] != p[1] ]
			if r:
				find_all_sequence(seq + [p], r)
			else:
				print(f"finished: {seq}")


def find_sequence(seq, remain):
	tail = remain[:]
	start = seq[-1]
	possibilities = [ p for p in tail if p[0] == start[1] ]
	if possibilities:
		p = random.choice(possibilities)
		r = [ q for q in tail if q[0] != p[0] or q[1] != p[1] ]
		if r:
			find_sequence(seq + [p], r)
		else:
			print(f"finished: {seq}")
	else:
		print('blocked')


def find_tailfirst(seq, remain):
	tail = remain[:]
	start = seq[-1]
	possibilities = [ p for p in tail if p[0] == start[1] ]
	if possibilities:
		p = possibilities[-1]
		r = [ q for q in tail if q[0] != p[0] or q[1] != p[1] ]
		if r:
			find_tailfirst(seq + [p], r)
		else:
			print('\n'.join([ f"{q[0]}-{q[1]}" for q in seq + [p]]))
	else:
		print('blocked')

def make_permutations(seq, remain):
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



n = 8
l = [ "white", "blue", "green", "yellow", "orange", "red", "purple", "gray", "black" ]
c = list(itertools.permutations(l, 2))
sequence = make_permutations([ c[0] ], c[1:])
print(len(sequence))

if sequence:
	for pair in sequence:
		print(f'{pair[0]} -> {pair[1]}\n')


