import sys
def main(cmd):
    try:
        runner = dict(
                fetch='main_fetch',
                show='main_show')[cmd]
    except KeyError:
        print >> sys.stderr, "Command not recognized:", cmd
        # XXX: Usage message
        sys.exit(1)

    module_tokens = ['datasets'] + sys.argv[1].split('.')
    # import as many as we can
    for i in range(len(module_tokens)):
        modname = '.'.join(module_tokens[:i+1])
        try:
            exec "import %s" % modname
        except ImportError:
            break
    # now carry on trying to interpret tokens as callable things
    # TODO
    # hail mary...
    fullname = "datasets.%s.%s" % (sys.argv[1], runner)
    try:
        exec "runner_fn = %s" % fullname
    except AttributeError:
        print >> sys.stderr, "Error: symbol '%s' not found" % fullname
        sys.exit(1)
    sys.exit(runner_fn())
