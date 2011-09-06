import main

def dataset_factory(identifier):
    # This function is simple now, but I think it would be OK to define a
    # calling convention so that identifier could be a dictionary or tuple
    # so that arguments could be passed to the constructor of the dataset
    # class.
    if isinstance(identifier, str):
        symbol = main.load_tokens(identifier.split('.'))
        return symbol()
    else:
        raise NotImplementedError()


