#!/bin/bash

export OMP_NUM_THREADS=1

# Copy all required files to Env
cp *.sh *.py ../Env && cd ../Env

# Run script
python main_mrf.py --reward-type="sparse" --lr=0.0001 --weight_predict=1.0  --pair_comp="bmm" --seed=0 --info="avg edges" --agent_type=-1

# Remove the files moved to env to keep env clean
rm -rf Agent.py Network.py arguments.py main_mrf.py misc.py runner.sh train_open_env.py utils.py

# Move resulting logs and parameters to original folder and go back to original folder
mv runs ../GPL-SPI && mv marlsave/parameters ../GPL-SPI
cd  ../GPL-SPI
