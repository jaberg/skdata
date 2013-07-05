
"""

This is a real-time dataset, reflecting the current situation on the streets of Austin TX.

No declared Dangerous Dog in the City of Austin and Travis County should ever be running at
large. They are court ordered to be restrained at all times and should be wearing a large tag
identifying them as a Dangerous Dog. They have attacked in the past. The owner is required to
provide $100,000 in financial responsibility. If they attack again the court could order them
put to sleep.

Data provided by: City of Austin

"""
import httplib
import json

root = "data.austintexas.gov"
dangerous_dogs_json = "/resource/ri75-pahg.json"

skdata_category = 'Public Safety'
skdata_tags = 'dangerous', 'dogs', 'public safety', 'pets', 'animals'

class DangerousDogs(object):
    """

    Attributes
    ----------
    `meta` is a list of dictionaries with keys:
        'first_name': dog owner's first name
        'last_name': dog owner's last name
        'address': dog owner's address
        'zip_code': dog owner's zip code
        'description_of_dog': free-form string, usually dog's name first
        'location': unclear, I'm guessing estimated location of dog.

    """
    def __init__(self):
        self.conn = httplib.HTTPConnection(root)
        self.conn.request("GET", dangerous_dogs_json)
        r1 = self.conn.getresponse()
        if r1.status == 200: # -- OK
            data1 = r1.read()
            self.meta = json.loads(data1)
        else:
            raise IOError('JSON resource not found', (r1.status, r1.reason))


