"""This file provides glumpy_viewer, a simple image-viewing mini-application.

The application is controlled via a state dictionary, whose keys are:

    'pos' - the current position in the image column we're viewing
    'window' - the glumpy window
    'I' - the glumpy.Image of the current column element
    'len' - the length of the image column

The application can be controlled by keys that have been registered with the
`command` decorator.  Some basic commands are set up by default:

    command('j') - advance the position
    command('k') - rewind the position
    command('0') - reset to position 0
    command('q') - quit

You can add new commands by importing the command decorator and using like this:

    >>> @command('r')
    >>> def action_on_press_r(state):
    >>>    ...          # modify state in place
    >>>    return None  # the return value is not used currently

The main point of commands right now is to update the current position
(state['pos']), in which case the window will be redrawn after the keypress
command returns to reflect the current position.

If you redefine a command, the new command clobbers the old one.

"""
import sys
import numpy as np
import glumpy


_commands = {}
def command(char):
    """
    Returns a decorator that registers its function for `char` keypress.
    """
    def deco(f):
        assert type(char) == str and len(char) == 1
        _commands[char] = f
        return f
    return deco


@command('j')
def inc_pos(state):
    state['pos'] = (state['pos'] + 1) % state['len']


@command('k')
def dec_pos(state):
    state['pos'] = (state['pos'] - 1) % state['len']


@command('0')
def reset_pos(state):
    state['pos'] = 0


@command('q')
def quit(state):
    sys.exit()


def glumpy_viewer(img_array,
        arrays_to_print = [],
        commands=None,
        cmap=None,
        window_shape=(512, 512),
        contrast_norm=None
        ):
    """
    Setup and start glumpy main loop to visualize Image array `img_array`.

    img_array - an array-like object whose elements are float32 or uint8
                ndarrays that glumpy can show.  larray objects work here.

    arrays_to_print - arrays whose elements will be printed to stdout
                      after a keypress changes the current position.

    """
    if contrast_norm not in (None, 'each', 'all'):
        raise ValueError('contrast_norm', contrast_norm)

    if contrast_norm == 'all':
        np.array(img_array, 'float32')
        img_array -= img_array.min()
        img_array /= max(img_array.max(), 1e-12)

    try:
        n_imgs = len(img_array)
    except TypeError:
        n_imgs = None

    state = dict(
            pos=0,
            fig=glumpy.figure((window_shape[1], window_shape[0])),
            I=glumpy.Image(img_array[0], colormap=cmap),
            len=n_imgs
            )

    fig = state['fig']
    if commands is None:
        commands = _commands

    @fig.event
    def on_draw():
        fig.clear()
        state['I'].draw(x=0, y=0, z=0,
                width=fig.width, height=fig.height)

    @fig.event
    def on_key_press(symbol, modifiers):
        if chr(symbol) not in commands:
            print 'unused key', chr(symbol), modifiers
            return

        pos = state['pos']
        commands[chr(symbol)](state)
        if pos == state['pos']:
            return
        else:
            img_i = img_array[state['pos']]
            if contrast_norm == 'each':
                # -- force copy
                img_i = np.array(img_i, 'float32')
                img_i -= img_i.min()
                img_i /= max(img_i.max(), 1e-12)

            #print img_i.shape
            #print img_i.dtype
            #print img_i.max()
            #print img_i.min()
            state['I'] = glumpy.Image(img_i,
                    colormap=cmap,
                    vmin=0.0,
                    vmax=1.0
                    )
            print state['pos'], [o[state['pos']] for o in arrays_to_print]
            fig.redraw()

    glumpy.show()

