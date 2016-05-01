#!/bin/bash
# Usage: create_dircopy source dest
# Will create in dest the directory tree of source
# source and dest must be absolute paths

curDir=$(pwd)
echo Starting copying script
cd ${curDir}/${1}
find . -type d > ${curDir}/dirs.txt
echo Made a copy of the folder
mkdir -p ${curDir}/${2}
cd ${curDir}/${2}
xargs mkdir -p < ${curDir}/dirs.txt
