#!/bin/bash

export OMP_NUM_THREADS=1
python main_mrf.py --lr=0.00025 --weight_predict=1.0  --pair_comp="bmm" --seed=0 --info="avg edges"
