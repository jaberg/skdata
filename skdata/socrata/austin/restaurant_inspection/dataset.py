
"""
Restaurant Inspection Scores

Provides restaurant scores for inspections performed within the last three
years. Online search of this data set also available at:
http://www.ci.austin.tx.us/health/restaurant/search.cfm


Data provided by: City of Austin

"""
import datetime
import httplib
import json

root = "data.austintexas.gov"
resource_json = "/resource/ecmv-9xxi.json"

skdata_tags = 'public safety', 'health', 'restaurants'

def do_casts(dct):
    rval = {
        'score': float(dct['score']),
        'restaurant_name': dct['restaurant_name'],
        'address': {
            'latitude': float(dct['address']['latitude']),
            'longitude': float(dct['address']['longitude']),
            'human_address': dct['address']['human_address'],
            'needs_recoding': bool(dct['address']['needs_recoding']),
            },
        'zip_code': dct['zip_code'],
        'inspection_date': datetime.datetime.fromtimestamp(
            int(dct['inspection_date'])),
        }
    return rval

class RestaurantInspectionScores(object):
    """

    Attributes
    ----------
    `meta` is a list of dictionaries with keys:
        'score': integer score <= 100
        'restaurant_name': string
        'address': dict
            'latitude': float
            'longitude': float
            'human_address': dict
                'address': string
                'city': string
                'zip': string
            'needs_recoding': bool
        'zip_code': string
        'inspection_date': date

    """
    def __init__(self):
        self.conn = httplib.HTTPConnection(root)
        self.conn.request("GET", resource_json)
        r1 = self.conn.getresponse()
        if r1.status == 200: # -- OK
            # XXX: retrieve *all* listings, not just first 1000 given by
            # default
            data1 = r1.read()
            self.meta = map(do_casts, json.loads(data1))
        else:
            raise IOError('JSON resource not found', (r1.status, r1.reason))

