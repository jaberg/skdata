from ..utils import get_my_path, xml2list, xml2dict
from os import path

MY_PATH = get_my_path()


def test_xml2list_voc07():
    gt = ['VOC2007',
          '000001.jpg',
          {'annotation': 'PASCAL VOC2007',
           'database': 'The VOC2007 Database',
           'flickrid': '341012865',
           'image': 'flickr'},
          {'flickrid': 'Fried Camels', 'name': 'Jinky the Fruit Bat'},
          {'depth': '3', 'height': '500', 'width': '353'},
          '0',
          {'bndbox': {'xmax': '195', 'xmin': '48', 'ymax': '371', 'ymin': '240'},
           'difficult': '0',
           'name': 'dog',
           'pose': 'Left',
           'truncated': '1'},
          {'bndbox': {'xmax': '352', 'xmin': '8', 'ymax': '498', 'ymin': '12'},
           'difficult': '0',
           'name': 'person',
           'pose': 'Left',
           'truncated': '1'}]
    xml_filename = path.join(MY_PATH, 'test.xml')
    gv = xml2list(xml_filename)
    assert gt == gv


def test_xml2dict_voc07():
    gt = {'filename': '000001.jpg',
          'folder': 'VOC2007',
          'object': [{'bndbox': {'xmax': '195',
                                 'xmin': '48',
                                 'ymax': '371',
                                 'ymin': '240'},
                      'difficult': '0',
                      'name': 'dog',
                      'pose': 'Left',
                      'truncated': '1'},
                     {'bndbox': {'xmax': '352',
                                 'xmin': '8',
                                 'ymax': '498',
                                 'ymin': '12'},
                      'difficult': '0',
                      'name': 'person',
                      'pose': 'Left',
                      'truncated': '1'}],
          'owner': {'flickrid': 'Fried Camels', 'name': 'Jinky the Fruit Bat'},
          'segmented': '0',
          'size': {'depth': '3', 'height': '500', 'width': '353'},
          'source': {'annotation': 'PASCAL VOC2007',
                     'database': 'The VOC2007 Database',
                     'flickrid': '341012865',
                     'image': 'flickr'}}
    xml_filename = path.join(MY_PATH, 'test.xml')
    gv = xml2dict(xml_filename)
    assert gt == gv
