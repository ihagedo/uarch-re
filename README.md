# uarch-re

Microbenchmarks that recover the hidden structure sizes of an out-of-order
core from timing alone.

```
. tools/env.sh
python3 -m unittest
./sweep.py toy --tag t --toy rob=64,int_prf=72,lq=24,sq=16 --n 2:90:2
```
