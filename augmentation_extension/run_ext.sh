#!/bin/sh
python3 -u exp_dd_extra.py > results2/dd_extra.log 2>&1
python3 -u exp_bv.py > results2/bv.log 2>&1
python3 -u exp_land.py > results2/land.log 2>&1
python3 -u exp_q1b.py > results2/q1b.log 2>&1
