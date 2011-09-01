

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
    dirname = path.dirname(tarName)
    if not path.exists(dirname):
        os.makedirs(dirname)
    tar_file = open(tarName, "wb+")
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
        print "Downloading '%s' to '%s'" % (url, tarName)
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
    if not path.exists(dirName):
        os.makedirs(dirName)
    for item in file_list:
        tar.extract(item, path=dirName)
        untar_size += 1
        status = r"Untared    : %20i Files [%4.1f%%]" % (untar_size,
                 untar_size * 100. / tar_size)
        status = status + chr(8) * (len(status) + 1)
        print status,
    print ''
    tar.close()

from urllib2 import urlopen
from tarfile import open as taropen
import os
from os import path

def download(url, output_dirname, overwrite=False):
    """Downloads file at `url` and write it in `output_dirname`"""

    basename = path.basename(url)
    page = urlopen(url)
    page_info = page.info()

    if not path.exists(output_dirname):
       os.makedirs(output_dirname)

    output_filename = path.join(output_dirname, basename)
    if path.exists(output_filename) and not overwrite:
        print("'%s' already exists! "
              "To overwrite, set overwrite=True."
              % output_filename)
        return
    output_file = open(output_filename, 'wb+')

    # size of the download unit
    block_size = 2**15
    dl_size = 0

    # display  progress only if we know the length
    print "Downloading '%s' to '%s'" % (url, output_filename)
    if 'content-length' in page_info:
        # file size in Kilobytes
        file_size = int(page_info['content-length']) / 1024.
        while True:
            buffer = page.read(block_size)
            if not buffer:
                break
            dl_size += block_size / 1024
            output_file.write(buffer)
            status = r"Downloaded : %10d Kilobytes [%4.1f%%]" \
                    % (dl_size, 100. * dl_size / file_size)
            status = status + chr(8) * (len(status) + 1)
            print status,
        print ''
    else:
        output_file.write(page.read())

    output_file.close()

#def extract(archive_filename, output_dirname, type='auto'):
    #pass

URL = "http://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz"
print download(URL, './tmp')
