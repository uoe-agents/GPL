# Open Ad Hoc Teamwork using Graph-based Policy Learning

This folder contains the implementation of the environments, experiments, and the GPL algorithm proposed in the **Open Ad Hoc Teamwork using Graph-based Policy Learning** paper. 
## Requirements

To install required packages, execute the following command:

```setup
pip install -r requirements.txt
```

We also require a modified version of OpenAI gym to run the provided codes. To do the necessary modifications to `gym`, check the directory of the `gym` package using

```setup
pip show gym
```

Assuming the package is installed in `<DIR>`, replace `<DIR>/gym/vector/async_vector_env.py` with the `async_vector_env.py` that we have provided. This can be achieved using the following command : 

```setup
cp async_vector_env.py <DIR>/gym/vector/async_vector_env.py
```


## Training

The codes for our experiments are all provided in `*_Experiments` folders. Each of these folders corresponds to the training code we used to train GPL and the baselines in a specific environment.
Before training the models, make sure to install the environments used in the experiments. For each environment, use the following instructions to install the environment:
```setup
cd env
pip install -e .
```

As an exception, the FortAttack experiments requires additional steps before starting the training process. For any approach that is being trained, the codes of that approach must be moved to the ```Env``` folder. This can be done by the following command :
```
mv <APPROACH FOLDER NAME> Env
```

Then, to train the algorithm/baseline for that experiment, run this command:

```train
./runner.sh
```

Full description of the hyperparameters and the architecture used in our work is provided in the appendix pdf file provided in the folder.

Aside from the component visualization experiment, periodic model checkpointing and evaluation has been included as part of the main script in the training implementation. We specifically run several episodes under the evaluation setup and log the resulting performance using tensorboard. However, since these logs can be quite big, we do not include it as part of the supplementary material. Instead, we include example csv result files that can be used to display the results presented in this work. 

## Visualization

The visualization codes are provided in the `Visualization` folder. For any result, you can compile the image by running the ```script_plot.py``` file.


With the GPL component visualization, the visualization code has already been provided as IPython notebook in `Experiments/component_visualization/Visualize.ipynb`. After executing `run.sh`, run the code from top to bottom to get a visualization of the different component models associated to GPL.

## Value Analysis
For joint action value analysis, you must first gather the joint action value data. You can do this by running,
```
mv Action_Value_Analysis/Data_Collection/* Open_Fortattack_Experiments/Env && cd Open_Fortattack_Experiments/Env
./runner.sh
```

This will results in several numpy arrays being stored in ```Open_Fortattack_Experiments/Env```. To do the analysis, you could then move these numpy arrays to the folder that contains the jupyter notebook for data analysis. You can do this by :
```
mv Open_Fortattack_Experiments/Env/*.npy Action_Value_Analysis/Data_Analysis

```

You can then run jupyter and execute the commands in the notebook.

## Pretrained policies
Due to file size restrictions for supplementary materials, we only include GPL-Q's pretrained models for FortAttack. This can be found in the ```PretrainedParameters``` folder. This parameter could then be loaded and used to control an agent by using the ```load()``` function in GPL-Q's ```Agent.py``` file.