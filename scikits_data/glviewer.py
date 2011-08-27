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


def glumpy_viewer(imgcol,
        other_cols_to_print = [],
        commands=None
        ):
    """
    Setup and start glumpy main loop to visualize Image Column `imgcol`.

    imgcol - a Column whose elements are float32 or uint8 ndarrays that glumpy
             can show.

    other_cols_to_print - other Columns whose elements will be printed to stdout
                          after a keypress changes the current position.

    """

    state=dict(
            window=glumpy.Window(512,512),
            pos=0,
            I=glumpy.Image(imgcol[0]),
            len=len(imgcol),
            )
    window = state['window']  # put in scope of handlers for convenience
    if commands is None:
        commands = _commands

    @window.event
    def on_draw():
        window.clear()
        state['I'].blit(0,0,window.width,window.height)

    @window.event
    def on_key_press(symbol, modifiers):
        if chr(symbol) not in commands:
            print 'unused key', chr(symbol), modifiers
            return

        pos = state['pos']
        commands[chr(symbol)](state)
        if pos == state['pos']:
            return
        else:
            state['I'] = glumpy.Image(imgcol[state['pos']])
            print state['pos'], [o[state['pos']] for o in other_cols_to_print]
            window.draw()


    window.mainloop()

