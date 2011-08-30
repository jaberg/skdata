import sys
def main(cmd):
    if 'fetch' == cmd:
        exec "import datasets.%s; sys.exit(datasets.%s.main_fetch())" % (
                sys.argv[1], sys.argv[1])
    if 'show' == cmd:
        exec "import datasets.%s; sys.exit(datasets.%s.main_show())" % (
                sys.argv[1], sys.argv[1])
    print >> sys.stderr, "Command not recognized:", cmd
    # XXX: Usage message
    sys.exit(1)
