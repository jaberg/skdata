"""
Provides a Dataset to access the nist digits dataset. 
"""

import os, numpy
from pylearn.io import filetensor as ft
from pylearn.datasets.config import data_root # config
from pylearn.datasets.dataset import Dataset

from pylearn.datasets.nist_sd import nist_to_float_11, nist_to_float_01


def load(dataset = 'train', attribute = 'data'):
  """Load the filetensor corresponding to the set and attribute.

  :param dataset: str that is 'train', 'valid' or 'test'
  :param attribute: str that is 'data' or 'labels'
  """
  fn = 'all_' + dataset + '_' + attribute + '.ft'
  fn = os.path.join(data_root(), 'nist', 'by_class', 'all', fn)

  fd = open(fn)
  data = ft.read(fd)
  fd.close()

  return data

def train_valid_test(ntrain=651668, nvalid=80000, ntest=82587, 
                     path=None, range = '01'):
  """
  Load the nist digits dataset as a Dataset.

  @note: the examples are uint8 and the labels are int32.
  @todo: possibility of loading part of the data.
  """
  rval = Dataset()

  # 
  rval.n_classes = 62
  rval.img_shape = (32,32)

  if range == '01':
    rval.preprocess = nist_to_float_01
  elif range == '11':
    rval.preprocess = nist_to_float_11
  else:
    raise ValueError('Nist Digits dataset does not support range = %s' % range)
  print "Nist Digits dataset: using preproc will provide inputs in the %s range." \
      % range

  # train
  examples = load(dataset = 'train', attribute = 'data')
  labels = load(dataset = 'train', attribute = 'labels')
  rval.train = Dataset.Obj(x=examples[:ntrain], y=labels[:ntrain])

  # valid
  rval.valid = Dataset.Obj(x=examples[651668:651668+nvalid], y=labels[651668:651668+nvalid])

  # test
  examples = load(dataset = 'test', attribute = 'data')
  labels = load(dataset = 'test', attribute = 'labels')
  rval.test = Dataset.Obj(x=examples[:ntest], y=labels[:ntest])
  
  return rval

