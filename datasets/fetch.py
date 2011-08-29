import sys
def main():
    exec "import datasets.%s; datasets.%s.main_fetch()" % (sys.argv[1], sys.argv[1])
