import argparse
import sys


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("mergedFile", help="source file")

    args = parser.parse_args(argv)

    bHeader = True
    mapPhyla = {}
    with open(args.mergedFile) as f:
    
        for line in f:
        
            line = line.rstrip()
            
            if bHeader:
                bHeader = False
                
                toks = line.split('\t')
                
                toks.pop(0)
                print(line)
            else:
                
                toks = line.split('\t')
                
                clade = toks[0]

                ctoks = clade.split('|')

                if len(ctoks) == 8:
                    
                    toks.pop(0)
                    tString = '\t'.join(toks)
                    mapPhyla[ctoks[7]] = '\t'.join(ctoks[:7])
                    print('%s\t%s' % (ctoks[7], tString))
    
    with open('phyla.tsv','w') as f:
    
        for scg,tax in mapPhyla.items():
            print('%s\t%s' % (scg,tax),file=f)

        



if __name__ == "__main__":
    main(sys.argv[1:])
