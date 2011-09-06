"""
Factory method for constructing datasets from string / dict / tuple.

"""

import main

def dataset_factory(identifier):
    """
    Return a dataset class instance based on a string or dictionary identifier.

    .. code-block:: python

        iris = dataset_factory('datasets.toy.Iris')

    This function works by parsing the string, and calling import and getattr a
    lot. (XXX)

    """
    # TODO: there is actually nothing dataset-specific in this implementation at
    # all, and in fact no guarantee that the return value is even a dataset.
    # This function simply implements a structured sort of 'exec' statement.
    #
    # This function is simple now, but I think it would be OK to define a
    # calling convention so that identifier could be a dictionary or tuple
    # so that arguments could be passed to the constructor of the dataset
    # class.
    if isinstance(identifier, str):
        symbol = main.load_tokens(identifier.split('.'))
        return symbol()
    elif isinstance(identifier, dict):
        raise NotImplementedError('dict API not defined yet', identifier)
    elif isinstance(identifier, (tuple, list)):
        raise NotImplementedError('seq API not defined yet', identifier)
    else:
        raise TypeError(identifier)


