from data_cache import cache_join, cache_open
import dataset
import column
import sys
import os

genders = ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'M']

def pubfig83_join(*names):
    return cache_join('pubfig', 'pubfig83', *names)

class pubfig83_blob(dataset.DatasetBlob):
    global_instance = None
    @classmethod
    def get_global_instance(cls):
        if cls.global_instance is None:
            cls.global_instance = cls()
        return cls.global_instance
        
    def fetch(self):
        dataset_dir = cache_join('pubfig')
        os.system('wget http://www.eecs.harvard.edu/~zak/pubfig83/pubfig83_first_draft.tgz -P ' + dataset_dir)
        os.system('cd ' + dataset_dir + '; tar -xzf pubfig83_first_draft.tgz')
        
    def load(self):
        dataset_path = pubfig83_join() 
        if not os.path.exists(dataset_path):
            self.fetch()
        names = os.listdir(dataset_path)
        names.sort()
        assert len(names) == len(genders)
        genders_names_pics = []
        ind = 0
        for gender, name in zip(genders, names):
            pics = os.listdir(pubfig83_join(name))
            pics.sort()
            for pic in pics:
                genders_names_pics.append(dict(
                    gender=gender,
                    name=name, 
                    id=ind,
                    jpgfile=pic))
                ind +=1 
                
        return genders_names_pics

class ImgRelPath(column.MapColumn):
    @staticmethod
    def _fn(dct):
        return ('pubfig', 'pubfig83', dct['name'], dct['jpgfile'])

class ImgFullPath(column.MapColumn):
    @staticmethod
    def _fn(dct):
        return pubfig83_join(dct['name'], dct['jpgfile'])

class PubFig83(dataset.Dataset):
    def __init__(self):
        self.blob = pubfig83_blob.get_global_instance()
        genders_names_pics = self.blob.load()
        columns = {}
        columns['meta'] = column.ListColumn(genders_names_pics)
        columns['gender'] = column.ListColumn(
                [g['gender'] for g in genders_names_pics])
        columns['name'] = column.ListColumn(
                [g['name'] for g in genders_names_pics])
        columns['jpgfile'] = column.ListColumn(
                [g['jpgfile'] for g in genders_names_pics])
        columns['img_relpath'] = ImgRelPath(columns['meta'])
        columns['img_fullpath'] = ImgFullPath(columns['meta'])
        columns['npy_img'] = column.NdarrayFromImagepath(columns['img_fullpath'])
        dataset.Dataset.__init__(self, columns)

def pubfig83_from_son(doc):
    return PubFig83()

if __name__ == '__main__':
    dset = PubFig83()
    dset.columns['npy_img'][0]
    if 0:
        print >> sys.stderr, img0.dtype
    from glviewer import glumpy_viewer, command
    
    glumpy_viewer(
            imgcol=dset.columns['npy_img'],
            other_cols_to_print=[
                dset.columns['name'],
                ])

