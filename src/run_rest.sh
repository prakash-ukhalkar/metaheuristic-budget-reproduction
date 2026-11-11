#!/bin/bash
# waits for the main sweep, then runs the remaining phases in order
cd "$(dirname "$0")"
until grep -q "PHASE COMPLETE" ../results/run_main.log; do sleep 30; done
for ph in "tune 32" "tune 128" "tuned 32" "tuned 128" "cons" "budget"; do
  python runner.py $ph > "../results/run_$(echo $ph | tr ' ' '_').log" 2>&1
done
echo ALLDONE > ../results/ALLDONE
