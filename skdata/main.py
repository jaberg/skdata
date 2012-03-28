"""
Entry point for bin/* scripts

XXX
"""
import sys
import logging
logger = logging.getLogger(__name__)

def import_tokens(tokens):
    # XXX Document me
    # import as many as we can
    rval = None
    for i in range(len(tokens)):
        modname = '.'.join(tokens[:i+1])
        # XXX: try using getattr, and then merge with load_tokens
        try:
            logger.info('importing %s' % modname)
            exec "import %s" % modname
            exec "rval = %s" % modname
        except ImportError, e:
            logger.info('failed to import %s' % modname)
            logger.info('reason: %s' % str(e))
            break
    return rval, tokens[i:]

def load_tokens(tokens):
    # XXX: merge with import_tokens
    logger.info('load_tokens: %s' % str(tokens))
    symbol, remainder = import_tokens(tokens)
    for attr in remainder:
        symbol = getattr(symbol, attr)
    return symbol

def main(cmd):
    """
    Entry point for bin/* scripts
    XXX
    """
    try:
        runner = dict(
                fetch='main_fetch',
                show='main_show',
                clean_up='main_clean_up')[cmd]
    except KeyError:
        print >> sys.stderr, "Command not recognized:", cmd
        # XXX: Usage message
        sys.exit(1)

    try:
        argv1 = sys.argv[1]
    except IndexError:
        logger.error('Module name required (XXX: print Usage)')
        return 1

    symbol = load_tokens(['skdata'] + argv1.split('.') + [runner])
    logger.info('running: %s' % str(symbol))
    sys.exit(symbol())
