import sys
import dataset

def fetch():
    vh = dataset.Calibrated()
    vh.fetch()


def show():
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
