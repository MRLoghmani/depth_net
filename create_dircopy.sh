#!/bin/bash
# Usage: create_dircopy source dest
# Will create in dest the directory tree of source
# source and dest must be absolute paths

cd ${1}
find . -type d > dirs.txt
mkdir -p ${2}
cd ${2}
xargs mkdir -p < ${1}/dirs.txt
