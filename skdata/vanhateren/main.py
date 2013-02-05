"""
Commands related to the van Hateren data set:

    python main.py fetch - download the data set (ctrl-C midway if you don't
        need the whole thing)

    python main.py show - show images from the data set using glumpy. Use 'j'
        and 'k' to move between images. For dependencies, type
        
        pip install glumpy && pip install pyopengl

    python main.py show_patches - show image patches from the data set using
        glumpy.  Use 'j' and 'k' to move between images. For dependencies, type
        
        pip install glumpy && pip install pyopengl
    
"""

import sys
import numpy as np
import dataset

def fetch():
    vh = dataset.Calibrated()
    vh.fetch()

def show():
    from skdata.utils.glviewer import glumpy_viewer
    vh = dataset.Calibrated(10)
    items = vh.meta[:10]
    images = np.asarray(map(vh.read_image, items))

    images = images.astype('float32')
    images /= images.reshape(10, 1024 * 1536).max(axis=1)[:, None, None]
    images = 1.0 - images

    glumpy_viewer(
            img_array=images,
            arrays_to_print=[items],
            window_shape=vh.meta[0]['image_shape'])

def show_patches():
    N = 100
    S = 128
    from skdata.utils.glviewer import glumpy_viewer
    vh = dataset.Calibrated(10)
    patches = vh.raw_patches((N, S, S), items=vh.meta[:10])

    patches = patches.astype('float32')
    patches /= patches.reshape(N, S * S).max(axis=1)[:, None, None]
    patches = 1.0 - patches


    SS = S
    while SS < 256:
        SS *= 2

    glumpy_viewer(
            img_array=patches,
            arrays_to_print=[vh.meta],
            window_shape=(SS, SS))

if __name__ == '__main__':
    sys.exit(globals()[sys.argv[1]]())
