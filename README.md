# Socratic Models - Team 2
| Sergei Agaronian, Theodoor Akkerboom, Maarten Drinhuyzen, Wenkai Pan, Marius Strampel

This Github is made for the course, Deep Learning 2 from the Universiteit van Amsterdam. With this project, a Socratic Model is made to predict the answer of the Raven Progressive Matrices(RPM). This is a visual IQ Test.

# How to run
1. Create the environment from the socrat.yml file:
```
conda env create -f socrat.yml
```
2. Activate the environment:
```
conda activate socrat
```
3. Run the main file:
```
python main.py

--name NAME                   How to name the results.
--seed SEED                   Seed to use for reproducing results.
--data_dir DATA_DIR           Data directory where to find dataset.
--split SPLIT                 Data split to use.
--type TYPE                   Puzzle type to use.
--ClassicOpenCV CLASSICOPENCV Use OpenCV or not
--vlm VLM                     VLM weights to use.
--lm LM                       LM weights to use.
```


# Structure
```
.
|
|-- center_single subset  # A subset of the RAVEN dataset for demos
|
|-- demos                 # Interactive notebooks showcasing the models
|   |-- Experiment1and2.ipynb
|   |-- experiment_3.ipynb
|
|-- output                # Output of the experiments
|   |-- experiment1
|   |-- experiment2
|
|-- src                   # Source code of the pipeline
|   |-- const.py     
|   |-- dataset.py
|   |-- model.py
|
|-- README.md   # Description of the repo with relevant getting started info (used dependencies/conda envs)
|
|-- blogpost.md # Blogpost style report
|
|-- lisa.job    # Job file for LISA cluster
|
|-- main.py     # Script to run the whole pipeline
|
|-- socrat.yml  # Environment
```
