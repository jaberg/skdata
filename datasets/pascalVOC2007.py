"""
Pascal VOC 2007 Dataset

website: http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/
"""

from urllib2 import urlopen
from tarfile import open as taropen
from xml.dom import minidom as dom
import os

if 0:
    # TODO: use get_cache_dir('pascalVOC2007')
    #       but only at object creation time, not import time.

    # importing the default 'PYTHOR3_DATA' environment
    # variable. That location will serve as a scratch
    # directory to store the dataset's images and raw
    # data.
    try:
        scratch = os.environ['PYTHOR3_DATA']
    except KeyError:
        raise KeyError('PYTHOR_DATA is not defined')


def download_and_untar(url, tarName, dirName):
    """
    **Description**
        open a url, expecting a tar file and downloads it
        to 'tarName'. Then untar the file inside 'dirName'

    **Parameters**
        url:
            valid URL link
        tarName:
            name of the tar file on the local machine to
            store the archive (full path)
        dirName:
            name of the directory into which one wants to
            untar the archive (full path)

    **Returns**
        nothing
    """
    #-- download part
    page = urlopen(url)
    tar_file = open(tarName, "wb")
    # size of the download unit (here 2**15 = 32768 Bytes)
    block_size = 32768
    dl_size = 0
    file_size = -1
    try:
        file_size = int(page.info()['content-length'])
    except:
        print "could not determine size of tarball so"
        print "no progress bar  on download displayed"
    if file_size > 0:
        while True:
            Buffer = page.read(block_size)
            if not Buffer:
                break
            dl_size += block_size
            tar_file.write(Buffer)
            status = r"Downloaded : %20d Bytes [%4.1f%%]" % (dl_size,
                     dl_size * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print status,
        print ''
    else:
        tar_file.write(page.read())
    tar_file.close()
    #-- untar part
    tar = taropen(tarName)
    file_list = tar.getmembers()
    untar_size = 0
    tar_size = len(file_list)
    for item in file_list:
        tar.extract(item, path=dirName)
        untar_size += 1
        status = r"Untared    : %20i Files [%4.1f%%]" % (untar_size,
                 untar_size * 100. / tar_size)
        status = status + chr(8) * (len(status) + 1)
        print status,
    print ''
    tar.close()


class PascalVOC2007(object):
    # by default, the 'temp' directory where the dataset is
    # being stored is the default 'PYTHOR3_DATA' environment
    # variable.
    def __init__(self, tmp_dir=scratch):
        self.tmp_dir = tmp_dir

    def fetch(self):
        """
        **Description**
            go fetch the Pascal VOC 2007 training, validation and testing
            data sets off the Internet. Then untar the archives in temporary
            directories.

        **Parameters**
            tmp_dir:
                optionnaly takes a string corresponding to the full path to the
                temp directory in which the data sets should be stored.

        **Returns**
            ``fetch`` will create two new attributes to the object
            self.trainvalDirName:
                full path to the temporary directory containing the training
                and validation data
            self.testDirName:
                full path to the temporary directory containing the testing
                data
        """

        # URL to the Pascal VOC 2007 training and validation
        # data set
        trainvalurl = "http://pascallin.ecs.soton.ac.uk/" + \
                      "challenges/VOC/voc2007/" + \
                      "VOCtrainval_06-Nov-2007.tar"
        # URL to the Pascal VOC 2007 testing data set
        testurl = "http://pascallin.ecs.soton.ac.uk/" + \
                  "challenges/VOC/voc2007/" + \
                  "VOCtest_06-Nov-2007.tar"
        # Names for the *.tar files
        trainvalTarName = os.path.join(self.tmp_dir, "trainval.tar")
        testTarName = os.path.join(self.tmp_dir, "test.tar")
        # Names for the directories in which to untar the files
        trainvalDirName = os.path.join(self.tmp_dir, "trainval")
        testDirName = os.path.join(self.tmp_dir, "test")
        # Actually downloading the tar files and untar
        dl.download_and_untar(trainvalurl, trainvalTarName, trainvalDirName)
        dl.download_and_untar(testurl, testTarName, testDirName)
        # assigns to the object two new attributes corresponding to the
        # full path to the trainval and test temporary directories
        self.trainvalDirName = trainvalDirName
        self.testDirName = testDirName

    def load(self):
        """
        **Description**
            Considering that the data has been fetched, this method will parse
            the Pascal VOC 2007 dataset and extract the metadata and assign it
            to the 'meta' attribute.

        **Parameters**
            uses the ``.trainvalDirName`` and ``.testDirName`` attributes
            which implies that the ``fetch`` method has been called already.

        **Returns**
            nothing. ``load`` just creates a list of meta data that it assigns
            to the ``.meta`` attribute of the object. Also, another attribute
            ``.fullpath`` will contain a list of all the images paths.
        """
        # path to the base directories
        try:
            trainval_dir = os.path.join(self.trainvalDirName, 'VOCdevkit',
                                        'VOC2007')
            test_dir = os.path.join(self.testDirName, 'VOCdevkit',
                                        'VOC2007')
        except AttributeError:
            raise AttributeError('did you fetch the data first ?')
        
        def _extract_filenames(self, pathname):
            """
            **Description**
                Given a path to a file containing images numbers, the function
                opens that file and returns a list of those numbers to which
                the suffix '.jpg' has been added.

            **Parameters**
                pathname:
                    full path to the file containing the image numbers

            **Returns**
                a list of file names of the type '002354.jpg'
            """
            if not os.path.exists(pathname):
                raise IOError('invalid path : %s' % pathname)
            elif not os.path.isfile(pathname):
                raise IOError('%s is not a file' % pathname)
            lines = open(pathname, 'r').readlines()
            return [line.split('\n')[0] + '.jpg' for line in lines]

        def _extract_meta(self, meta_paths, identities):
            """
            **Description**
                extracts the meta data from a list of XML files (given by
                'meta_paths'). What's more, in order to find the identity
                of the image file (i.e. whether it is a 'train', 'val' or
                'test' type of data), we use the dictionnary 'identities'.

            **Parameters**
                meta_paths:
                    a list of paths to all the XML files containing the meta
                    data information
                identities:
                    a dictionnary containing a set of three keys 'train',
                    'val' and 'test'. For each of those keys, there is an
                    associated list of filenames of the form '005342.jpg'
                    which correspond to the image filenames for that key.

            **Returns**
                a list of dictionnaries ``[{...},{...},...]`` which constitute
                the meta data for the Pascal VOC 2007 dataset.
            """
            meta = []
            fullpath = []
            global_size = len(meta_paths)
            local_size = 0
            for path in meta_paths:
                meta_info = {}
                try:
                    lines = open(path, 'r').readlines()
                    xml_file = dom.parseString(''.join(lines))
                except:
                    raise IOError('could not parse %s' % path)
                # extract all the fields from the XML source
                meta_info['folder'] = \
                xml_file.getElementsByTagName('folder')[0].firstChild.wholeText
                meta_info['filename'] = \
                xml_file.getElementsByTagName('filename')[0].firstChild.wholeText
                meta_info['segmented'] = \
                bool(xml_file.getElementsByTagName('segmented')[0].firstChild.wholeText)
                source = xml_file.getElementsByTagName('source')
                meta_info['source'] = {
                'database' :
                source[0].getElementsByTagName('database')[0].firstChild.wholeText,
                'annotation' :
                source[0].getElementsByTagName('annotation')[0].firstChild.wholeText,
                'image' :
                source[0].getElementsByTagName('image')[0].firstChild.wholeText,
                'flickrid' :
                source[0].getElementsByTagName('flickrid')[0].firstChild.wholeText
                }
                owner = xml_file.getElementsByTagName('owner')
                meta_info['owner'] = {
                'flickrid' :
                owner[0].getElementsByTagName('flickrid')[0].firstChild.wholeText,
                'name' :
                owner[0].getElementsByTagName('name')[0].firstChild.wholeText
                }
                size = xml_file.getElementsByTagName('size')
                meta_info['size'] = {
                'width' :
                int(size[0].getElementsByTagName('width')[0].firstChild.wholeText),
                'height' :
                int(size[0].getElementsByTagName('height')[0].firstChild.wholeText),
                'depth' :
                int(size[0].getElementsByTagName('depth')[0].firstChild.wholeText),
                }
                objects = xml_file.getElementsByTagName('object')
                list_of_objects = []
                for item in objects:
                    to_add = {}
                    to_add['name'] = \
                    item.getElementsByTagName('name')[0].firstChild.wholeText
                    to_add['pose'] = \
                    item.getElementsByTagName('pose')[0].firstChild.wholeText
                    to_add['truncated'] = \
                    bool(item.getElementsByTagName('truncated')[0].firstChild.wholeText)
                    to_add['difficult'] = \
                    bool(item.getElementsByTagName('difficult')[0].firstChild.wholeText)
                    bndbox = item.getElementsByTagName('bndbox')[0]
                    to_add['bndbox'] = {
                    'xmin' : \
                    int(bndbox.getElementsByTagName('xmin')[0].firstChild.wholeText),
                    'ymin' : \
                    int(bndbox.getElementsByTagName('ymin')[0].firstChild.wholeText),
                    'xmax' : \
                    int(bndbox.getElementsByTagName('xmax')[0].firstChild.wholeText),
                    'ymax' : \
                    int(bndbox.getElementsByTagName('ymax')[0].firstChild.wholeText)
                    }
                    list_of_objects.append(to_add)
                meta_info['objects'] = list_of_objects
                # finally add the 'split' key which tells whether the
                # image belongs to the 'train', 'val' or 'test' set.
                # we also compute the full path to the image file and
                # append it to the fullpath list.
                if meta_info['filename'] in identities['test']:
                    meta_info['split'] = 'test'
                    suffix = meta_info['filename']
                    fullpath.append(
                             os.path.join(self.testDirName, 'VOCdevkit',
                             'VOC2007', 'JPEGImages', suffix) )
                elif meta_info['filename'] in identities['train']:
                    meta_info['split'] = 'train'
                    suffix = meta_info['filename']
                    fullpath.append(            
                             os.path.join(self.trainvalDirName, 'VOCdevkit',
                             'VOC2007', 'JPEGImages', suffix) )
                else:
                    meta_info['split'] = 'val'
                    suffix = meta_info['filename']
                    fullpath.append(                          
                             os.path.join(self.trainvalDirName, 'VOCdevkit',
                             'VOC2007', 'JPEGImages', suffix) )
                # appends the whole meta_info dictionnary to meta
                meta.append(meta_info)
                # update progress bar
                local_size += 1
                status = r"Extracted  : %20i Files [%4.1f%%]" % (local_size,
                         local_size * 100. / global_size)
                status = status + chr(8) * (len(status) + 1)
                print status,
            print ''
            return (meta, fullpath)

        # first we map each image filename to an identity corresponding
        # to 'train', 'val' or 'test' and gather the information into a
        # dictionnary
        identity = {}
        trainval_prefix = os.path.join(trainval_dir, 'ImageSets', 'Main')
        test_prefix = os.path.join(test_dir, 'ImageSets', 'Main')
        for suffix in ['train.txt', 'val.txt', 'test.txt']:
            if suffix == 'train.txt':
                pathname = os.path.join(trainval_prefix, suffix)
                identity['train'] = _extract_filenames(self, pathname)
            elif suffix == 'val.txt':
                pathname = os.path.join(trainval_prefix, suffix)
                identity['val'] = _extract_filenames(self, pathname)
            elif suffix == 'test.txt':
                pathname = os.path.join(test_prefix, suffix)
                identity['test'] = _extract_filenames(self, pathname)
        # the meta data is contained in directory 'Annotations'
        # for both the testing data and training/validation data
        trainval_prefix = os.path.join(trainval_dir, 'Annotations')
        test_prefix = os.path.join(test_dir, 'Annotations')
        trainval_meta_file_paths = [os.path.join(trainval_prefix, suffix) for
                                    suffix in os.listdir(trainval_prefix)]
        test_meta_file_paths = [os.path.join(test_prefix, suffix) for suffix in
                                    os.listdir(test_prefix)]
        meta_file_paths = trainval_meta_file_paths + test_meta_file_paths
        (self.meta, self.fullpath) = _extract_meta(self, meta_file_paths, identity)

    def erase_load(self):
        """
        if a pickle of the meta data is on disk, this function
        erases it.
        """
        raise NotImplementedError('erase_load not implemented')

    def erase_fetch(self):
        """
        In the case of Pascal VOC 2007, erases the *.tar archives
        """
        raise NotImplementedError('erase_fetch not implemented')
