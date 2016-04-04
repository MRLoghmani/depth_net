import os,time
import subprocess
import sys,getopt
from subprocess import call

def main(argv):
	start_time=time.time()
	sourcedir = ''
	outputdir = ''
   	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'depth_tree.py -i <sourcedir> -o <outputdir>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print 'usage: depth_tree3_0.py -i <source_rootdir> -o <output_rootdir>'
			print 'example: python depth_tree3_0.py -i /home/prusso/Blenders/Francesca -o /home/hades/basefolder'
	 		print 'hint: the source_rootdir needs to have all the blend folders inside it: source_rootdir/object1/ob1.blend'
			sys.exit()
		elif opt in ("-i", "--ifile"):
	 		sourcedir = arg
		elif opt in ("-o", "--ofile"):
	 		outputdir = arg
	print 'Input root dir is "', sourcedir
	print 'Output root dir is "', outputdir
	for root, dirs, filenames in os.walk(sourcedir):
		filenames.sort()
	   	for f in filenames:
			extension = os.path.splitext(f)[1]
	   		#import code
	   		#code.interact(local=locals())
			if extension == '.blend':
				fullfilepath = "\""+os.path.abspath(os.path.join(root, f))+"\""		
			   	callstring="blender"+" -b "+fullfilepath+" --python"+" depth_all3_0.py"+ " -- " + fullfilepath + " " + outputdir
			   	subprocess.call(callstring,shell=True)
	print("traversed all the tree in: %d ",time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])
