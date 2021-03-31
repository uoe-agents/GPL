#!/bin/bash

export OMP_NUM_THREADS=1

# Copy all required files to Env
cp -r $1 ../../Open_Experiments/FortAttack/Env
cp *.sh *.py ../../Open_Experiments/FortAttack/Env && cd ../../Open_Experiments/FortAttack/Env

# Run data collection code
python test_open_env.py --reward-type="sparse" --lr=0.0001 --weight_predict=1.0  --pair_comp="bmm" --seed=0 --info="avg edges" --loading-dir=$1

# Remove the files moved to env to keep env clean
rm -rf Agent.py Network.py arguments.py main_mrf.py misc.py runner.sh train_open_env.py utils.py $1

mv *.npy ../../../Action_Value_Analysis/Data_Analysis
cd  ../GPL