import sys
import dataset

def fetch():
    vh = dataset.NaturalImages()
    vh.fetch()

if __name__ == '__main__':
    sys.exit(globals()[sys.argv[1]]())
