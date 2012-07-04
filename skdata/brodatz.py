"""
Brodatz texture dataset.

http://www.ux.uis.no/~tranden/brodatz.html

"""

import hashlib
import logging
import os
import urllib

from PIL import Image

from .data_home import get_data_home
from .utils.image import ImgLoader
from .larray import lmap


logger = logging.getLogger(__name__)
url_template = 'http://www.ux.uis.no/~tranden/brodatz/D%i.gif'

valid_nums = range(1, 113)
del valid_nums[13]


class Brodatz(object):
    """
    self.meta is a list of dictionaries with the following structure:

    - id: a unique number within the list of dictionaries
    - basename: relative filename such as "D1.gif"
    - image: sub-dictionary
        - shape: (<height>, <width>)
        - dtype: 'uint8'

    """

    DOWNLOAD_IF_MISSING = True

    def home(self, *names):
        return os.path.join(get_data_home(), 'brodatz', *names)

    @property
    def meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=self.DOWNLOAD_IF_MISSING)
            self._meta = self.build_meta()
            return self._meta

    def fetch(self, download_if_missing=None):
        if download_if_missing is None:
            download_if_missing = self.DOWNLOAD_IF_MISSING

        if download_if_missing:
            if not os.path.isdir(self.home()):
                os.makedirs(self.home())

        sha1s = sha1_list.split('\n')

        for ii, image_num in enumerate(valid_nums):
            url = url_template % image_num
            dest = self.home(os.path.basename(url))
            if not os.path.exists(dest):
                if download_if_missing:
                    logger.warn("Downloading ~100K %s => %s" % (url, dest))
                    downloader = urllib.urlopen(url)
                    data = downloader.read()
                    tmp = open(dest, 'wb')
                    tmp.write(data)
                    tmp.close()
                else:
                    raise IOError(dest)
            sha1 = hashlib.sha1(open(dest).read()).hexdigest()
            if sha1 != sha1s[ii + 1]:
                raise IOError('SHA1 mismatch on image %s', dest)

    def build_meta(self):
        meta = []
        for i, image_num in enumerate(valid_nums):
            basename = 'D%i.gif' % image_num
            try:
                img_i = Image.open(self.home(basename))
            except:
                logger.error('failed to load image %s' % self.home(basename))
                raise
            meta.append(dict(
                    id=i,
                    basename=basename,
                    image=dict(
                        shape=img_i.size,
                        dtype='uint8',
                        )
                    ))
        return meta

    @classmethod
    def main_fetch(cls):
        return cls().fetch(download_if_missing=True)

    @classmethod
    def main_clean_up(cls):
        return cls().clean_up()

    def images_larray(self, dtype='uint8'):
        img_paths = [self.home(m['basename']) for m in self.meta]
        imgs = lmap(ImgLoader(ndim=2, dtype=dtype, mode='L'),
                           img_paths)
        return imgs

    @classmethod
    def main_show(cls):
        from utils.glviewer import glumpy_viewer, command, glumpy
        self = cls()
        imgs = self.images_larray('uint8')
        Y = range(len(imgs))
        glumpy_viewer(
                img_array=imgs,
                arrays_to_print=[Y],
                cmap=glumpy.colormap.Grey)


def gen_sha1_list():
    ds = Brodatz()
    foo = open('foo.txt', 'w')
    for image_num in valid_nums:
        data = open(ds.home('D%i.gif' % image_num)).read()
        sha1 = hashlib.sha1(data).hexdigest()
        print >> foo, sha1


def main_fetch():
    Brodatz.main_fetch()


def main_show():
    Brodatz.main_show()


def main_clean_up():
    Brodatz.main_clean_up()


if __name__ == '__main__':
    gen_sha1_list()


sha1_list = """
6aea21c25826a22222045befb90e5def42040cc1
ff2ee9e834e61e30c5cb915a34ad4fdbfd736cf7
7ac47673659ddcea3143bb90e766dc145ca45bf6
1b0ede375d2a19ca61d8343d428b4f72be747a0f
7ffb4161c6c78742e970cef1f264fe69151304a1
1e9c45897662d6e9f238b0c101f686e581de9aca
0e45e15a3031bd36b5e5272e943dfaad06c4a886
36c3a413a357e10b0462a2e7eeaa57a4b489312f
0036b3196a6d3e84bc43c31a0f9d355340dd4359
5de79b9f56fbae5cd6373045ed32a3aa31480599
e7f1c262256ac00fa08cb5de6f9a3eb8a6547408
a499f68f8b2345cd4f1ad1a07220187928a46aea
105b82cb4ff1f115b799ff51d6f220d6667e2cff
83c618339db659dcfe5152153adfdfc8e6fedf76
7d409e860116934df6acbd301ee715b02addeb57
246684aa363923b9d429a08b84ec00fb84a9e4e5
e23792018a6053c77e639dd604330d590611c311
5f8d4e7667b2119dbc60c87b6037a17165509a63
31b16441452138edf84f532afa0c1a7180d62fb4
c2afb40c2915d535bf253a74db6070b71e2edbe7
08e2c583ec90b7f0d8441cc5f6a1aa80a8d41248
b68a1ace83d438aba08473263db24d4ec9ae0f21
60a52085468e3abb98467282812ab3be4b9ff2f4
0786200ab65e4e301b64eb93ec2f066a2a82da8b
c99d7b85365f2ff2e76b53390c6165595b2168f2
ae5903d5b4e1ee2c420cd45f321fe3d19cfe118f
a69b1ef0cc1cd4be39602965ebc7a53ddd75b0a6
bf5733046ab98caaa430a8907c99ca90ce9e7788
5c1b8b0cf5abef47659dea3a863537dd8805321f
a7e74d916024d2c9438d2ed5396c9bbe9afaff9b
dc42adf47e3df5902e1405487745c28a14424416
0a8e7c7a18cdea2548c46684be707a3374209784
c19afb81855571984422cf13daa8bd24d5001d7d
f5e17d13897fa116c22c77a99b6c712c8effd865
7f60c98fa6a017f95f55eb34fd8edcc3f0e8dc5c
7f5a021572c602d11428d352b9f67493873c0efc
9199ac4d864287928925eea0c22e39a4d3baa88a
ba49f85a54727ede2adf767b7920fe73a22127c2
a6c53c51087face6ed82978c3da4ce3f26faa0c9
dc34603d037031af73a28d83ebfb8fbca5915f5b
607029253bdc179e52fb98be6f37174724da14f8
a8bac24d2abee2eb918451ccdc6068eaa96b9107
798ec414d36574ccf70644e6d7817f19d5487a68
e510dae0ac04c0777a7d6ba1c7cf2704f6241e14
0ea65d7e1c02a64059c940b70ce6cc9cd769aac7
ab078bf1f3aa323326696724bde2a4da326b7934
befca82ece334e44ea22a01c2c607289dc124235
f81b8faa5b6f9d1b4b7f36a3b31a6e8577ee98b9
1df2ce25ae49892c8989a6769160af0c1dfb97c7
e97b97334b29539db7bd37005dabc9866ec93b29
53de157b2c6ed794da12dabe8d7b8992fb272eef
666b0e98b7065acc20165eadc2cd3524ff29215e
c5dacf7c2d96e38bda0122241705dcd2ca597276
8cf86108974c390601dfa0a43dd266b3b548d2e4
c8caac53a4eaf3eb1369fb4e91d034fc120671bc
f38777eb82b382ce98ba97baa27801504634ba87
cdafce0f6a481b7ed545daf6e9fd8b1b0cf62668
4d65bc3f718f935a231efcd6cb0c227cfc60f3d0
cfe5243b07a2d97e30a211a4e97ffbade8f317e8
fecdf27729853a98f2b5108429167f3dbc62b4eb
f06bb13968c7a08f5698d4a2e0b41dbcf30ccda9
ab683e55d4c3d47b4fc0ca347514ac157b907d3b
1a9ba577d5a3600d3f1c002644bc55d6a1800e10
e39c4aa064ed20880a52111a887745470adc6779
913ca78db851f7b17380eacf11d80546a34b3106
1a63e33d4db18c3b0838e2f63a9d64cd087260d6
07245841bca7eb9fada1f3ed4baecd875561430d
30d6d58fd1ba2580e5006d64edfac8bf8ca89db6
e27bb5e24813b8f107cf20445fced07e466beb94
3e99f3c00f65adee54e8208086a4a1edb6719bcd
47eca012a7b878c730bc588d27c877602b6ca557
903f4fb23c7f3999d55c724c2a0549f00e022ac5
502933bd77c9b956ebb5a463f7000f6795cb1d8a
1b1895d0a08a96dafe93d08de19b41269e171b8e
3959cf92cb01f59219f028bf1be2997a943b8d51
03d2962a642e989dfd94d09800083f8f2972beb0
98ee9dc3ba7062759ef0715edf1ddbc8598c20f2
c1d8816c4b3c6c38a3bfff649d232dd72cf18138
277da2112d65d25c18e444fa129f80887e2c7389
cfc24d04acd1ec1f766654ffb10a451ea7fc0b51
3eadd78b4994fb8c1f99bd8fe2031afe46545014
8daa87856f87d5f7eaca082951df9ec53643b76b
defdb079e6101390f57360c812efc5b635e2b9e1
2fcf58f0527bdcf55328ebe47fa2b6d53152dbda
48cdb4014c75283de1b1ba8c8ac25622c3086881
f02edcc7a7a9639d92538e925d6f66a20aa9b9d8
fd4b399d54edc14782db726a79e161e149a24444
a32607c4f0a414205fce6d8c83a942a30031107c
aa17150ca98d659827d49d006d6faa642ba8d578
500f8749556ad44e4ec9d76c7b05dd2bfe183d71
120065d86a2c2c96c18a07d2d73379406b31d11c
c95232ef22c988eeb1bec35493da056fbb97550e
e336c1bca6815291231d67e8e152ab18027b8f58
ca709c085e71b2aa79a04c0c62ff4ed4d85b79f1
2066566e70a55ac3f10046825ec77a652a16401a
3edd31e7a2764f97be19712a339c7e7d51b0dca9
279644656faf50b569ed778f9701b8a104655ac0
d0e3b068368d98e4081199a90aaa174bbc96d2ae
a86a410a1498ab7d535bf8f675dc6c1855cec2a0
75b9636f0a2db93d4394e03fd02e87bb926d6ef6
d57e5a296d0fc2858bdf61f5f8ab97b924a8a288
5c428868ba39275a4b55c43235ca6e6ec5cfdcb7
423d727f3bfd62a1f66942ff27d2ffa90d7595fd
b308c2a346b6321acff87bee927ba2ae38e26b43
9431fba01bf3c9c1b711d69fc2167418084e36bb
95b8d95cd6287c9f581fd783fe5513a867a0b18d
29ee33df0d4df5fdabd2c94eb80f5b98587ec5f4
dbb601aaf26f4b93e32643c03d594495958de962
6ffff10c8e8959ddd8c5d35b6d95a55f2537df78
db9911e79ddf3984304a5c5fad1c9720f79334ce
b2e01384421741e899c850f9144609a80d7f0c46
"""

