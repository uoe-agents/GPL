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

The codes for our open ad hoc teamwork experiments are provided in `Open_Experiments` folder. The `Open_Experiments` folder contains three folders, each containing the environment, GPL-Q, and GPL-SPI implementation for a specific environment used in our work.
Before training the models in LBF and Wolfpack, make sure to install the environments used in the experiments using the following commands:
```setup
cd Open_Experiments/<Environment Name>/env
pip install -e .
```

For all environments, run this command to train GPL-Q or GPL-SPI :

```train
cd Open_Experiments/<Environment Name>/<Approach Name>
./runner.sh
```

Full description of the hyperparameters and the architecture used in our work is provided in the appendix of our work. 

Aside from training a GPL-Q of GPL-SPI model, the shell script also periodically checkpoints the model and evaluates it in the training and evaluation environment. We specifically run several episodes under the evaluation setup and log the resulting performance using tensorboard. The resulting logs can be viewed using the following command : 

```
tensorboard --logdir=Open_Experiments/<Environment Name>/<Approach Name>/runs
```

Using the displayed logs, we can see the different metrics we reported in our work such as the average total returns per episode and the shooting accuracy of the learner. To compile the logs of the experiments into the plots we included in our work, you can download the logs as a csv and use the visualization codes in the `Visualization` folder.

## Visualization

The visualization codes are provided in the `Visualization` folder. For any result, you can compile the image by running the ```vis.py``` file.