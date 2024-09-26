# Predicting cellular injury

[![DOI](https://zenodo.org/badge/744169074.svg)](https://zenodo.org/doi/10.5281/zenodo.12514972)

![workflow](./notebooks/4.visualization/figures/workflow_fig.png)
> Diagram illustrating the steps taken in this study. We first downloaded the cell-injury dataset from GitHub and the Image Data Resource (IDR) to obtain the raw morphological features and associated metadata. Using Pycytominer, we processed these features to generate feature-selected profiles, which were then split into training, testing, and holdout sets for model training. Finally, we applied our trained model to the JUMP dataset to predict cellular injuries for previously unseen compounds.

The goal of this project was to use [Pycytominer](https://github.com/cytomining/pycytominer) to generate feature-selected profiles from image-based data and train a multi-class logistic regression model to predict cellular injury.

We sourced the cell-injury dataset from the [IDR](https://idr.openmicroscopy.org/webclient/?show=screen-3151) and its corresponding [GitHub repository](https://github.com/IDR/idr0133-dahlin-cellpainting).
Using [Pycytominer](https://github.com/cytomining/pycytominer), we processed these datasets to prepare them for model training.
We then trained our model to predict 15 different types of injuries using the cell-injury dataset and applied the trained model to the JUMP dataset to identify cellular injuries.

## Data sources

We obtained the cell-injury dataset from the [IDR](https://idr.openmicroscopy.org/webclient/?show=screen-3151) and its associated [GitHub repository](https://github.com/IDR/idr0133-dahlin-cellpainting).
After processing these datasets with [Pycytominer](https://github.com/cytomining/pycytominer) to prepare them for model training, we trained a model to predict 15 different types of injuries using the cell-injury dataset.
We then applied this trained model to the JUMP dataset to predict cellular injuries.

| Data Source | Description |
|-------------|-------------|
| [IDR repository](https://github.com/IDR/idr0133-dahlin-cellpainting/tree/main/screenA) | Repository containing annotated screen data |
| [Supplemental metadata](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36829-x/MediaObjects/41467_2023_36829_MOESM5_ESM.xlsx) | Supplementary metadata dataset |
| [JUMP](https://jump-cellpainting.broadinstitute.org/) | JUMP data repository|

## Repository structure

Overall structure of our repo.

| Directory | Description |
|-----------|-------------|
| [data](./data) | Contains all the datasets |
| [results](./results) | Stores all results generated from the notebook modules in the `./notebooks` directory |
| [src](./src) | Contains utility functions |
| [notebooks](./notebooks/) | Contains all notebook modules for our analysis |

## Notebook structure

Below are all the notebook modules used in our study.

| Notebook | Description |
|----------|-------------|
| [0.feature_selection](./notebooks/0.feature_selection/) | Labels wells with cellular injuries, identifies shared features between the cell injury dataset and JUMP, and selects features for the cell injury dataset |
| [1.data_splits](./notebooks/1.data_splits/) | Generates training and testing splits, including holdout sets (plate, treatment, and random wells) |
| [2.modeling](./notebooks/2.modeling/) | Trains and evaluates a multi-class logistic regression model |
| [3.jump_analysis](./notebooks/3.jump_analysis/) | Applies our model to the JUMP dataset to predict cellular injuries |
| [4.visualizations](./notebooks/4.visualizations/) | Contains a notebook responsible for generating our figures |

## Installing repository and dependencies

This installation guide assumes that you have Conda installed.
If you do not have Conda installed, please follow the documentation [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. **Clone the repository**: Clone the repository into your local machine and change the directory to the repo.

    ```bash
    git clone git@github.com:WayScience/Cytotoxic-Nuisance-Metadata-Analysis.git && cd Cytotoxic-Nuisance-Metadata-Analysis
    ```

2. **Install dependencies**: There are two environment files in this repository.
One is in the root project folder, containing all the necessary packages to conduct our analysis in Python.
The other is within the `./notebooks/4.visualization` directory, which includes packages for generating plots with R.

    - **Analysis environment**: Create and activate the environment for analysis.

        ```bash
        conda env create -f cell_injury.yaml
        conda activate cell-injury
        ```

    - **R visualization environment**: Create and activate the environment for R visualization.

        ```bash
        conda env create -f ./notebooks/4.visualization/visualization_env.yaml
        conda activate visualization-env
        ```

That's it! Your Conda environments should now be set up with the specified packages from the YAML files.

## How to use the notebook modules

The notebooks are meant to be run in sequence, with each module representing a distinct step in the process.
Each module also includes a `.sh` script to automate the execution of its corresponding notebooks.

All notebook results are stored in the `./results` directory, which is organized into subfolders numbered according to the module that generated the results.

For example, if you want to run the `1.data_splits` module (assuming that you have already completed the previous module `0.feature_selection_and_data/`), you can follow these steps:

1. Navigate to the module directory:

   ```bash
   cd notebooks/1.data_splits
   ```

2. Execute the shell script to run the notebook code:

   ```bash
   source ./1.data_splits.sh
   ```

## Analysis summary

### Feature selection

Before performing feature selection, we first labeled the wells associated with each injury.
We did this using datasets from the [cell-injury study](https://www.nature.com/articles/s41467-023-36829-x), which provided details on treatments linked to specific injuries.
After mapping the injury labels to the wells based on their treatments, we proceeded with feature alignment.

We identified the features shared between the `cell-injury` dataset and the JUMP dataset, focusing only on these "shared" morphological features.

We then applied feature selection using [Pycytominer](https://github.com/cytomining/pycytominer) to extract the most informative and least redundant features.
This process produced our `feature-selected` profiles, which were subsequently used to train the multi-class logistic regression model.

### Data splitting

We split our data into training, testing, and holdout sets.
First, we generated our holdout splits, which are reserved exclusively for evaluating the model's performance and are not used during the model's training or tuning.

1. **Plate Holdout Set**: We randomly selected 10 plates.
2. **Treatment Holdout Set**: We selected one treatment per injury to hold out.
However, given that some cellular injuries have very low diversity in treatments, we created a heuristic that only holds out treatments if the cell injuries have more than 10 different treatments.
For cell injuries that didn't meet this criterion, no treatments were held out.
3. **Well Holdout Set**: We randomly selected wells in each plate.
Due to a significant imbalance between control wells and treated wells, we selected a fixed number of 5 control wells and 10 randomly selected treated wells.

All holdout datasets were removed from the main dataset.
The remaining data was then split into training and testing sets using an 80/20 train/test split, respectively.
We used the `stratify` parameter to preserve the distribution of labels, preventing label imbalances that could lead to classification problems.

### Training the model

We trained a multi-class logistic regression model using randomized cross-validation for hyperparameter tuning to fine-tune our model.
Our logistic regression model was set to `multi_class="multinomial"`, indicating that it applies a softmax approach to handle and classify multiple classes.
The model also included `class_weight="balanced"` to ensure the model pays more attention to minority classes during training, preventing label imbalance or bias.

### Applying model to JUMP data

We applied our trained multi-class logistic regression model to the JUMP dataset to predict injuries present in each well.
We loaded the JUMP dataset and selected only the features that were shared with the `cell-injury` dataset.
Using the model, we generated predictions for each well and then analyzed these predictions to understand the distribution and types of injuries.

### Generating figures

We used R to generate the figures, which requires switching to a specific `conda` environment that contains all the necessary dependencies.
All the files used to generate these figures can be found in the `./results` directory.
To replicate the figures, follow the steps below (assuming you are in the project's root directory):

1. Navigate to the visualization module:

   ```bash
   cd ./notebooks/4.visualization/
   ```

2. Execute the bash script that sets up the Conda environment and runs the R scripts to generate the figures:

   ```bash
   source ./4.visualization.sh
   ```

This script will automatically create the required environment and execute the necessary R code.
Once the process completes, all figures will be saved in the `./figures` directory.

## Development Tools Used

In our project, we utilized a variety of development tools to enforce coding standards, formatting, and improve overall code quality.
Below is a list of the primary technologies and linters used:

- [**pre-commit**](https://github.com/pre-commit/pre-commit): A framework for managing and maintaining multi-language pre-commit hooks. This ensures that code meets the necessary quality standards before committing changes to the repository.
- [**pycln**](https://github.com/hadialqattan/pycln): A tool to automatically remove unused imports from Python files, keeping the codebase clean and optimized.
- [**isort**](https://github.com/PyCQA/isort): An import sorting tool that organizes imports according to a specific style (in this case, aligned with Black's formatting rules). This helps maintain consistency in the order of imports throughout the codebase.
- [**ruff-pre-commit**](https://github.com/astral-sh/ruff-pre-commit): A fast Python linter and formatter that checks code style and can automatically fix formatting issues.
- [**blacken-docs**](https://github.com/adamchainz/blacken-docs): A utility that formats Python code within documentation blocks, ensuring that example code snippets in docstring and markdown files adhere to the same standards as the main codebase.
- [**pre-commit-hooks**](https://github.com/pre-commit/pre-commit-hooks): A collection of various hooks, such as removing trailing whitespace, fixing end-of-line issues, and formatting JSON files.

**Note:** to see the pre-commit configurations please refer to the `./.pre-commit-config.yaml` file

For machine learning, we used:

- [**scikit-learn (sklearn)**](https://github.com/scikit-learn/scikit-learn): A robust and widely-used machine learning library that provides simple and efficient tools for data analysis and modeling.

## Citing our work

Please consider using our paper to cite our work or the usage of Pycytominer.
> Erik Serrano, Srinivas Niranj Chandrasekaran, Dave Bunten, Kenneth I. Brewer, Jenna Tomkinson, Roshan Kern, Michael Bornholdt, Stephen Fleming, Ruifan Pei, John Arevalo, Hillary Tsang, Vincent Rubinetti, Callum Tromans-Coia, Tim Becker, Erin Weisbart, Charlotte Bunne, Alexandr A. Kalinin, Rebecca Senft, Stephen J. Taylor, Nasim Jamali, Adeniyi Adeboye, Hamdah Shafqat Abbasi, Allen Goodman, Juan C. Caicedo, Anne E. Carpenter, Beth A. Cimini, Shantanu Singh, Gregory P. Way <https://arxiv.org/abs/2311.13417>
