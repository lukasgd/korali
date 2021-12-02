#!/bin/bash

rm -rf _deps

# clone CUP
git clone -b amr2 --recursive git@github.com:cselab/CUP2D.git _deps/CUP-2D

make -C _deps/CUP-2D/makefiles -j

