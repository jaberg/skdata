"""
Copyright (c) 2010 Gary Wilson Jr. <gary.wilson@gmail.com> and contributers.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

From:
http://pypi.python.org/pypi/python-archive/0.1
http://code.google.com/p/python-archive/

Changelog:
----------

* 2011/09/02: Cosmetic changes and add verbose kwarg to {Tar,Zip}Archive
classes by Nicolas Pinto <pinto@rowland.harvard.edu>

"""

import os
import tarfile
import zipfile


class ArchiveException(Exception):
    """Base exception class for all archive errors."""


class UnrecognizedArchiveFormat(ArchiveException):
    """Error raised when passed file is not a recognized archive format."""


def extract(archive_filename, output_dirname='./', verbose=True):
    """
    Unpack the tar or zip file at the specified `archive_filename` to the
    directory specified by `output_dirname`.
    """
    Archive(archive_filename).extract(output_dirname, verbose=verbose)


class Archive(object):
    """
    The external API class that encapsulates an archive implementation.
    """

    def __init__(self, file):
        self._archive = self._archive_cls(file)(file)

    @staticmethod
    def _archive_cls(file):
        cls = None
        if isinstance(file, basestring):
            filename = file
        else:
            try:
                filename = file.name
            except AttributeError:
                raise UnrecognizedArchiveFormat(
                    "File object not a recognized archive format.")
        base, tail_ext = os.path.splitext(filename.lower())
        cls = extension_map.get(tail_ext)
        if not cls:
            base, ext = os.path.splitext(base)
            cls = extension_map.get(ext)
        if not cls:
            raise UnrecognizedArchiveFormat(
                "Path not a recognized archive format: %s" % filename)
        return cls

    def extract(self, output_dirname='', verbose=True):
        self._archive.extract(output_dirname, verbose=verbose)

    def list(self):
        self._archive.list()


class BaseArchive(object):
    """
    Base Archive class.  Implementations should inherit this class.
    """

    def extract(self):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError


class ExtractInterface(object):
    """
    Interface class exposing common extract functionalities for
    standard-library-based Archive classes (e.g. based on modules like tarfile,
    zipfile).
    """

    def extract(self, output_dirname, verbose=True):
        if not verbose:
            self._archive.extractall(output_dirname)
        else:
            members = self.get_members()
            n_members = len(members)
            for mi, member in enumerate(members):
                self._archive.extract(member, path=output_dirname)
                extracted = mi + 1
                status = (r"Progress: %20i files extracted [%4.1f%%]"
                          % (extracted, extracted * 100. / n_members))
                status += chr(8) * (len(status) + 1)
                print status,
            print


class TarArchive(ExtractInterface, BaseArchive):

    def __init__(self, filename):
        self._archive = tarfile.open(filename)

    def list(self, *args, **kwargs):
        self._archive.list(*args, **kwargs)

    def get_members(self):
        return self._archive.getmembers()


class ZipArchive(ExtractInterface, BaseArchive):

    def __init__(self, filename):
        self._archive = zipfile.ZipFile(filename)

    def list(self, *args, **kwargs):
        self._archive.printdir(*args, **kwargs)

    def get_members(self):
        return self._archive.namelist()


extension_map = {
    '.egg': ZipArchive,
    '.jar': ZipArchive,
    '.tar': TarArchive,
    '.tar.bz2': TarArchive,
    '.tar.gz': TarArchive,
    '.tgz': TarArchive,
    '.tz2': TarArchive,
    '.zip': ZipArchive,
}
