# Installation
`conda create --name probnes python=3.11`\
`conda activate probnes`\
`conda install pip`\
`pip install numpy matplotlib evotorch==0.5.1 gpytorch==1.13 botorch pyyaml imageio emukit ucimlrepo pybnn glfw gym geoopt`\

# Usage
Experiments are defined through configurations files in `\config`. Configuration examples for each of the experiments produced in the paper are available directly. If parameters are inserted as a list, the cartesian product of all parameters entered as a list will be executed sequentially.

Once a configuration file is defined, simply execute the command `python main.py --config {config_name}` where `{config_name}.yaml` is your configuration file to run the defined experiments.

The results of the experiments will be available in the folder `\logs`.

To produce the figures displayed in the paper, simply use the command `python plot_scripts.py --config {config_name}`. The optional argumment `--log True` will make the plot in log scale.



