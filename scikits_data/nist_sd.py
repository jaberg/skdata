"""
Provides a Dataset to access the nist digits_reshuffled dataset. 
"""

import os, numpy
from pylearn.io import filetensor as ft
from pylearn.datasets.config import data_root # config
from pylearn.datasets.dataset import Dataset

def nist_to_float_11(x):
  return (x - 128.0)/ 128.0

def nist_to_float_01(x):
  return x / 255.0

def load(dataset = 'train', attribute = 'data'):
  """Load the filetensor corresponding to the set and attribute.

  :param dataset: str that is 'train', 'valid' or 'test'
  :param attribute: str that is 'data' or 'labels'
  """
  fn = 'digits_reshuffled_' + dataset + '_' + attribute + '.ft'
  fn = os.path.join(data_root(), 'nist', 'by_class', 'digits_reshuffled', fn)

  fd = open(fn)
  data = ft.read(fd)
  fd.close()

  return data

def train_valid_test(ntrain=285661, nvalid=58646, ntest=58646, path=None,
    range = '01'):
  """
  Load the nist reshuffled digits dataset as a Dataset.

  @note: the examples are uint8 and the labels are int32.
  @todo: possibility of loading part of the data.
  """
  rval = Dataset()

  # 
  rval.n_classes = 10
  rval.img_shape = (32,32)

  if range == '01':
    rval.preprocess = nist_to_float_01
  elif range == '11':
    rval.preprocess = nist_to_float_11
  else:
    raise ValueError('Nist SD dataset does not support range = %s' % range)
  print "Nist SD dataset: using preproc will provide inputs in the %s range." \
      % range

  # train
  examples = load(dataset = 'train', attribute = 'data')
  labels = load(dataset = 'train', attribute = 'labels')
  rval.train = Dataset.Obj(x=examples[:ntrain], y=labels[:ntrain])

  # valid
  examples = load(dataset = 'valid', attribute = 'data')
  labels = load(dataset = 'valid', attribute = 'labels')
  rval.valid = Dataset.Obj(x=examples[:nvalid], y=labels[:nvalid])

  # test
  examples = load(dataset = 'test', attribute = 'data')
  labels = load(dataset = 'test', attribute = 'labels')
  rval.test = Dataset.Obj(x=examples[:ntest], y=labels[:ntest])
  
  return rval


