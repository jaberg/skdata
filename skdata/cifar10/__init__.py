"""
CIFAR-10 Image classification dataset

Data available from and described at:
http://www.cs.toronto.edu/~kriz/cifar.html

If you use this dataset, please cite "Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""

import dataset

def main_fetch():
    dataset.CIFAR10().fetch(True)


def main_show():
    self = dataset.CIFAR10()
    from utils.glviewer import glumpy_viewer, glumpy
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=CIFAR10._pixels,
            arrays_to_print=[Y],
            cmap=glumpy.colormap.Grey,
            window_shape=(32 * 2, 32 * 2))


def main_clean_up():
    dataset.CIFAR10().clean_up()
