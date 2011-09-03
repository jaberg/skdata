"""Helpers to download and extract archives"""

# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          Nicolas Poilvert <poilvert@rowland.harvard.edu>
# License: BSD 3 clause

from urllib2 import urlopen
from os import path

import archive

def download(url, output_filename):
    """Downloads file at `url` and write it in `output_dirname`"""

    page = urlopen(url)
    page_info = page.info()

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
            status = r"Progress: %20d kilobytes [%4.1f%%]" \
                    % (dl_size, 100. * dl_size / file_size)
            status = status + chr(8) * (len(status) + 1)
            print status,
        print ''
    else:
        output_file.write(page.read())

    output_file.close()


def extract(archive_filename, output_dirname, verbose=True):
    """Extracts `archive_filename` in `output_dirname`.

    Supported archives:
    -------------------
    * Zip formats and equivalents: .zip, .egg, .jar
    * Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2
    """
    print "Extracting '%s' to '%s'" % (archive_filename, output_dirname)
    archive.extract(archive_filename, output_dirname, verbose=verbose)


def download_and_extract(url, output_dirname, verbose=True):
    """Downloads and extracts archive in `url` into `output_dirname`.

    Note that `output_dirname` has to exist and won't be created by this
    function.
    """
    archive_basename = path.basename(url)
    archive_filename = path.join(output_dirname, archive_basename)
    download(url, archive_filename)
    extract(archive_filename, output_dirname, verbose=verbose)
