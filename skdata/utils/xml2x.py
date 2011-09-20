# WARNING: this module and its functions/objects are not bulletproof and they
# may fail to deliver the expected results in some situations, use at your own
# risk!


def xml2dict(xml_filename):
    tree = ElementTree.parse(xml_filename)
    root = tree.getroot()
    xml_dict = XmlDictConfig(root)
    return xml_dict


def xml2list(xml_filename):
    tree = ElementTree.parse(xml_filename)
    root = tree.getroot()
    xml_list = XmlListConfig(root)
    return xml_list

# -----------------------------------------------------------------------------
# Modified from http://code.activestate.com/recipes/410469-xml-as-dictionary
# -----------------------------------------------------------------------------
from xml.etree import ElementTree


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if len(element):
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):

        children_names = [child.tag for child in parent_element.getchildren()]

        if parent_element.items():
            self.update(dict(parent_element.items()))

        for element in parent_element:

            if len(element):

                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    child_dict = XmlDictConfig(element)

                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    child_dict = {element[0].tag: XmlListConfig(element)}

                # if the tag has attributes, add those to the dict
                if element.items():
                    child_dict.update(dict(element.items()))

                if children_names.count(element.tag) > 1:
                    if element.tag not in self:
                        # the first of its kind, an empty list must be created
                        self[element.tag] = [child_dict]
                    else:
                        self[element.tag] += [child_dict]
                else:
                    self.update({element.tag: child_dict})

            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})

            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})
