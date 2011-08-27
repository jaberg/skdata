""" Functions related to the datasets used in Larochelle et al. 2007 (incl. modified MNIST). 

These datasets were introduced in
"An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation"
Hugo Larochelle, Dumitru Erhan, Aaron Courville, James Bergstra and Yoshua Bengio. In Proc of 
International Conference on Machine Learning (2007).

url:
http://www.iro.umontreal.ca/~lisa/twiki/pub/Public/DeepVsShallowComparisonICML2007/icml-2007-camera-ready.pdf

"""
import os, sys
import numpy

from data_cache import get_cache_dir

class DatasetLoader(object):
    """
    A class for loading an ICML07 dataset into memory.

    The class has functionality to 
    - download the dataset from the internet  (in amat format)
    - convert the dataset from amat format to npy format
    - load the dataset from either amat or npy source files
    """
    def __init__(self, http_source, 
            n_inputs, n_classes,
            n_train, n_valid, n_test,
            npy_filename_root, 
            amat_filename_root=None, 
            amat_filename_train=None,
            amat_filename_test=None,
            amat_filename_all=None,
            ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def download(self, todir):
        #TODO: write a system call to wget to dl the file from self.http_source
        raise NotImplementedError()

    def load_from_amat(self):
        if self.amat_filename_all is not None:
            amat_all = AMat(self.amat_filename_all)
            allmat = amat_all.all
            assert allmat.shape[0] == self.n_train + self.n_valid + self.n_test, allmat.shape
        else:
            if self.amat_filename_root is not None:
                amat_train = AMat(self.amat_filename_root+'_train.amat')
                amat_test = AMat(self.amat_filename_root+'_test.amat')
            else:
                amat_train = AMat(self.amat_filename_train)
                amat_test = AMat(self.amat_filename_test)
            assert amat_train.all.shape[0] == self.n_train + self.n_valid
            assert amat_test.all.shape[0] == self.n_test
            allmat = numpy.vstack((amat_train.all, amat_test.all))
        # CHECKPOINT: allmat has been computed by this point.
        assert allmat.shape[1] == self.n_inputs+1
        inputs = allmat[:, :self.n_inputs].astype('float32')
        labels = allmat[:, self.n_inputs].astype('int8')
        assert numpy.allclose(labels, allmat[:, self.n_inputs])
        assert numpy.all(labels < self.n_classes)
        return inputs, labels

    def load_from_amat_save_to_numpy(self):
        inputs, labels = self.load_from_amat()
        numpy.save(self.npy_filename_root+'_inputs.npy', inputs)
        numpy.save(self.npy_filename_root+'_labels.npy', labels)
        return inputs, labels

    def load_from_numpy(self, mmap_mode='r'):
        """Much faster than load_from_amat"""
        inputs = numpy.load(self.npy_filename_root+'_inputs.npy', mmap_mode=mmap_mode)
        labels = numpy.load(self.npy_filename_root+'_labels.npy', mmap_mode=mmap_mode)
        assert inputs.shape == (self.n_train + self.n_valid + self.n_test, self.n_inputs)
        assert labels.shape[0] == inputs.shape[0]
        assert numpy.all(labels < self.n_classes)
        return inputs, labels

def icml07_loaders(new_version=True, rootdir='.'):
    rval = dict(
        mnist_basic=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip',
            amat_filename_root=os.path.join(rootdir, 'mnist'),
            npy_filename_root=os.path.join(rootdir, 'mnist_basic'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            ),
        mnist_background_images=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip',
            amat_filename_root=os.path.join(rootdir, 'mnist_background_images'),
            npy_filename_root=os.path.join(rootdir, 'mnist_background_images'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            ),
        mnist_background_random=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip',
            amat_filename_root=os.path.join(rootdir, 'mnist_background_random'),
            npy_filename_root=os.path.join(rootdir, 'mnist_background_random'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            ),
        rectangles=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip',
            amat_filename_root=os.path.join(rootdir, 'rectangles'),
            npy_filename_root=os.path.join(rootdir, 'rectangles'),
            n_inputs=784,
            n_classes=10,
            n_train=1000,
            n_valid=200,
            n_test=50000
            ),
        rectangles_images=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip',
            amat_filename_root=os.path.join(rootdir, 'rectangles_im'),
            npy_filename_root=os.path.join(rootdir, 'rectangles_images'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            ),
        convex=DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip',
            amat_filename_root=os.path.join(rootdir, 'convex'),
            npy_filename_root=os.path.join(rootdir, 'convex'),
            n_inputs=784,
            n_classes=10,
            n_train=6500, #not sure about this train/valid split
            n_valid=1500,
            n_test=50000
            ),
        )
    for level in range(1,7):
        rval['mnist_noise_%i'%level] = DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_noise_variation.tar.gz',
            amat_filename_all=os.path.join(rootdir,
                'mnist_noise_variations_all_%i.amat'%level),
            npy_filename_root=os.path.join(rootdir, 'mnist_noise_%i'%level),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=2000
            )

    if new_version:
        rval['mnist_rotated'] = DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip',
            amat_filename_test=os.path.join(rootdir,
                'mnist_all_rotation_normalized_float_test.amat'),
            amat_filename_train=os.path.join(rootdir,
                'mnist_all_rotation_normalized_float_train_valid.amat'),
            npy_filename_root=os.path.join(rootdir, 'mnist_rotated'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            )
        rval['mnist_rotated_background_images'] = DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip',
            amat_filename_test=os.path.join(rootdir,
                'mnist_all_background_images_rotation_normalized_test.amat'),
            amat_filename_train=os.path.join(rootdir,
                'mnist_all_background_images_rotation_normalized_train_valid.amat'),
            npy_filename_root=os.path.join(rootdir, 'mnist_rotated_background_images'),
            n_inputs=784,
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            )
    else:
        raise NotImplementedError('TODO: what are the amat_filenames here')
        rval['mnist_rotated'] = DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation.zip')
        rval['mnist_rotated_background_images'] = DatasetLoader(
            http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image.zip')
    return rval


