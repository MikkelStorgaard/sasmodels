#!/bin/sh

echo "./main.py dab,dab_futhark -nq=5000 -neval=100 -double -noplot"
./main.py dab,dab_futhark -nq=5000 -neval=100 -double -noplot
echo "./main.py dab,dab_futhark -nq=5000 -neval=100 -double -noplot -2d"
./main.py dab,dab_futhark -nq=5000 -neval=100 -double -noplot -2d
echo
echo "./main.py line,line_futhark -nq=5000 -neval=100 -double -noplot"
./main.py line,line_futhark -nq=5000 -neval=100 -double -noplot
echo "./main.py line,line_futhark -nq=5000 -neval=100 -double -noplot -2d"
./main.py line,line_futhark -nq=5000 -neval=100 -double -noplot -2d
echo
echo "./main.py broad_peak,broad_peak_futhark -nq=5000 -neval=100 -double -noplot"
./main.py broad_peak,broad_peak_futhark -nq=5000 -neval=100 -double -noplot
echo "./main.py broad_peak,broad_peak_futhark -nq=5000 -neval=100 -double -noplot -2d"
./main.py broad_peak,broad_peak_futhark -nq=5000 -neval=100 -double -noplot -2d
