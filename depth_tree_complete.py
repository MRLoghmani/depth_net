import os,time
import subprocess
import sys,getopt
from subprocess import call
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser(
        description='usage: depth_tree_complete.py --input <source_rootdir> --output <output_rootdir> --tag <string>')
    parser.add_argument("--output")
    parser.add_argument("--input")
    parser.add_argument("--tag", help="Default is empty", default='')
    parser.add_argument("--n_samples",type=int, help="Number of samples per instance", default=60)
    args = parser.parse_args()
    return args


def main(argv):
        args=get_arguments()
	start_time=time.time()
	sourcedir = ''
	outputdir = ''
        sourcedir = args.input
        outputdir = args.output
        tag = args.tag
        n_samples=args.n_samples
	print 'Input root dir is "', sourcedir
	print 'Output root dir is "', outputdir
	print 'tag is"', tag
	for root, dirs, filenames in os.walk(sourcedir):
		filenames.sort()
	   	for f in filenames:
			extension = os.path.splitext(f)[1]
	   		#import code
	   		#code.interact(local=locals())
			if extension == '.blend':
				fullfilepath = "\""+os.path.abspath(os.path.join(root, f))+"\""		
			   	callstring="/home/poseidon/blender-2.77a-linux-glibc211-x86_64/./blender"+" -b "+fullfilepath+" --python"+" depth_all_complete.py"+ " -- " + fullfilepath + " " + outputdir + " " + tag + " %d" % (n_samples)
			   	subprocess.call(callstring,shell=True)
	print("traversed all the tree in: %d ",time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])

