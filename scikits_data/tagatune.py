"""
Access the TagATune dataset from

http://tagatune.org/Datasets.html
"""

from __future__ import absolute_import

import os
import numpy

import theano

from .config import data_root

def read_annotations_final(path):
    """Return a parsed (column-wise) representation of the tagatune/annotations_final.csv file
    
    :param path: an openable string to locate the file tagatune/annotations_final.csv

    :returns: 4-tuple (list of clip_ids, list of attribute lists, list of mp3 paths, list of
    attribute names)

    """
    f = open(path)
    attributes = []
    mp3_paths = []
    clip_ids = []
    for line_idx, line in enumerate(f):
        if line_idx == 0:
            #this is the header line, it contains all the column names
            column_names = [eval(tok) for tok in line[:-2].split('\t')]
            assert len(column_names) == 190
            assert column_names[0] == 'clip_id'
            assert column_names[-1] == 'mp3_path'
        else:
            #strip the leading and trailing '"' symbol from each token
            column_values = [tok[1:-1] for tok in line[:-2].split('\t')]
            assert len(column_values) == 190
            clip_ids.append(int(column_values[0]))
            mp3_paths.append(column_values[-1])
            # assert we didn't chop off too many chars
            assert column_values[-1].endswith('.mp3')
            attributes_this_line = column_values[1:-1]

            # assert that the data is binary
            assert all(c in '01' for c in attributes_this_line)
            attributes.append(numpy.asarray([int(c) for c in attributes_this_line],
            dtype='int8'))

    # assert that we read all the lines of the file
    assert len(clip_ids) == 25863
    assert len(attributes) == 25863
    assert len(mp3_paths) == 25863

    attribute_names = column_names[1:-1] #all but clip_id and mp3_path
    return clip_ids, attributes, mp3_paths, attribute_names

def cached_read_annotations_final(path):
    if not hasattr(cached_read_annotations_final, 'rval'):
        cached_read_annotations_final.rval = {}
    if not path in cached_read_annotations_final.rval:
        cached_read_annotations_final.rval[path] = read_annotations_final(path)
    return cached_read_annotations_final.rval[path]

def test_read_annotations_final():
    return read_annotations_final(data_root() + '/tagatune/annotations_final.csv')

class TagatuneExample(theano.Op):
    """
    input - index into tagatune database (not clip_id)
    output - clip_id, attributes, path to clip's mp3 file
    """
    def __init__(self, music_dbs='/data/gamme/data/music_dbs'):
        self.music_dbs = music_dbs
        annotations_path = music_dbs + '/tagatune/annotations_final.csv'
        self.clip_ids, self.attributes, self.mp3_paths, self.attribute_names =\
                cached_read_annotations_final(annotations_path)

    n_examples = property(lambda self: len(self.clip_ids))

    def make_node(self, idx):
        _idx = theano.tensor.as_tensor_variable(idx, ndim=0)
        return theano.Apply(self, 
                [_idx], 
                [theano.tensor.lscalar('clip_id'),
                    theano.tensor.bvector('clip_attributes'),
                    theano.generic('clip_path')])
    def perform(self, node, (idx,), out_storage):
        out_storage[0][0] = self.clip_ids[idx]
        out_storage[1][0] = self.attributes[idx]
        out_storage[2][0] = self.music_dbs + '/tagatune/clips/mp3/' + self.mp3_paths[idx]

    def grad(self, inputs, output):
        return [None for i in inputs]

#tagatune_example = TagatuneExample() #requires reading a big data file
