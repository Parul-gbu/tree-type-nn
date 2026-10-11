#!/bin/sh
# Full pipeline (single CPU core, roughly 40-50 min). Edit the paths at the top of prepare_data.py first.
python3 -u prepare_data.py
python3 -u run_main.py            > results/logs/main.log 2>&1
python3 -u receptive_fields.py    > results/logs/rf.log 2>&1
python3 -u rf_control.py          > results/logs/rf_control.log 2>&1
python3 -u double_descent.py      > results/logs/dd.log 2>&1
python3 -u double_descent_refined.py > results/logs/dd_refined.log 2>&1
python3 -u size_min.py            > results/logs/size_min.log 2>&1
python3 -u size_fair.py           > results/logs/size_fair.log 2>&1
