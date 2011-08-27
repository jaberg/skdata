"""
Load the MajorMiner dataset
"""

import logging, os,sys
from .config import data_root
_logger = logging.getLogger('pylearn.datasets.majorminer')

def three_column(tagfile=None, trackroot=None, expected_tagfile_len=51556):
    """Load meta-information of major-miner dataset

    Data is stored as a three-column file:

        <tag> <count> <mp3 path>

    This function returns the parsed file as a list of 3-tuples.
    
    """
    if tagfile is None:
        tagfile = os.path.join(data_root(), 'majorminer', 'three_column.txt')
        _logger.info('Majorminer loading %s'%tagfile)

    if trackroot is None:
        trackroot = os.path.join(data_root(), 'majorminer')
        _logger.info('Majorminer using trackroot %s'%tagfile)

    tag_count_track = []

    for line in open(tagfile):
        if line:
            tag, count, track = line[:-1].split('\t')
            tag_count_track.append((tag, int(count), os.path.join(trackroot, track)))

    if expected_tagfile_len:
        if len(tag_count_track) != expected_tagfile_len:
            raise Exception('Wrong number of files listed')

    return tag_count_track

try:
    import mad
except ImportError:
    pass

def remove_bad_tracks(three_col, min_seconds=8):
    """Heuristically filter the three_col data to contain only valid tracks
    """
    bad_tracks = set()
    all_tracks = set()

    silent_tracks = []
    missing_in_action = []
    too_short = []

    try:
        _file = mad.MadFile
        test_len = True
    except:
        _file = file
        test_len = False


    for tag, count, track in three_col:
        if track in all_tracks:
            continue
        all_tracks.add(track)
        if tag in set(['silence', 'end', 'nothing']):
            bad_tracks.add(track)
            silent_tracks.append(track)
            _logger.debug("silent file: %s" % track)
            continue

        try:
            t = _file(track)
        except IOError:
            bad_tracks.add(track)
            missing_in_action.append(track)
            _logger.debug("missing file: %s"% track)
            # it is normal to have 2
            #if len(missing_in_action) > 5:
                #raise Exception('Too many missing files:', missing_in_action)
            continue

        if test_len and t.total_time() < min_seconds*1000:
            # too short
            bad_tracks.add(track)
            _logger.debug("short file: %f %s" %(t.total_time(), track))
            too_short.append((track, t.total_time()))
            # it is normal to have maybe 10?
            #if len(too_short) > 40:
                #raise Exception('Too many short files:', too_short)
            continue

    if silent_tracks:
        _logger.warning("removed %i silent files"% len(silent_tracks))

    if missing_in_action:
        _logger.warning("missing %i files"% len(missing_in_action))

    if too_short:
        _logger.warning("discarded %i files less than %f seconds long"%(
            len(too_short), min_seconds))

    _logger.info("kept %i of %i tracks"% (len(all_tracks)-len(bad_tracks),
        len(all_tracks)))

    # return a cleaned three_column list
    rval = []
    for tag, count, track in three_col:
        if track not in bad_tracks:
            rval.append((tag, count, track))
    return rval



def list_tracks(three_col):
    tracks = list(set(tup[2] for tup in three_col))
    tracks.sort()
    return tracks

def list_tags(three_col):
    tags = list(set(tup[0] for tup in three_col))
    tags.sort()
    return tags

def track_tags(three_col, tracks, tags):
    """Return the count of each tag for each track
    [ [(tag_id, count), (tag_id, count), ...],   <---- for tracks[0]
      [(tag_id, count), (tag_id, count), ...],   <---- for tracks[1]
      ...
    ]
    """
    tag_id = dict(((t,i) for i,t in enumerate(tags)))
    track_id = dict(((t,i) for i,t in enumerate(tracks)))
    rval = [[] for t in tracks]
    for tag, count, track in three_col:
        rval[track_id[track]].append((tag_id[tag], count))
    return rval



class Meta(object):
    def __init__(self, tagfile=None, trackroot=None, expected_tagfile_len=51556,
            filter_broken=True):
        self.three_column = three_column(tagfile, trackroot, expected_tagfile_len)
        if filter_broken:
            self.three_column = remove_bad_tracks(self.three_column)
        self.tracks = list_tracks(self.three_column)
        self.tags = list_tags(self.three_column)
        self.track_tags = track_tags(self.three_column, self.tracks, self.tags)

        _logger.info('MajorMiner meta-information: %i tracks, %i tags' %(
            len(self.tracks), len(self.tags)))

        #for tt in self.track_tags:
        #    print tt

