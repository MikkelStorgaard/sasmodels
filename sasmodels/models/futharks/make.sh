#!/bin/sh
for f in $(find -type f -name "*.fut" -printf "%f\n"); do
  futhark-pyopencl --library $f
done
