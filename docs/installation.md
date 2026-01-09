# Installation

Back to main page: [README](../README.md)

## 1. Install python dependencies

Create a new conda environment if necessary. 

```bash
conda create -n [env_name] python=3.11
```

Note: if you wish to do statistical tests using the R-Python bridge (rpy2), the python version in the environment must be <=3.11.
If python=3.11, you can install rpy2 version 3.6+.

```bash
pip install -r requirements.txt
```
or
```bash
pip install --user -r requirements.txt
```


## 2. Huggingface setup.
- Set up huggingface token to allow downloading of model/dataset.
- Export token to HF_TOKEN (in ~/.bashrc):
```
export HF_TOKEN={your_token}
```

You might need to create a cache folder and set up environment var to prevent 'disk quota exceeded' error:
```bash
mkdir -p /scratch/{your_username}/huggingface_cache
export HF_HOME=/scratch/{your_username}/huggingface_cache
```

## 3. File management
- In src/constants.py, specify the filepath to store the dataset.
- Also specify where to get OOD data. Will probably automate this at some point.
```
if USER == "joeltjy1":
    JORDAN_DATASET_FILEPATH = "/scratch/joel/jordan_dataset"
    OOD_DATASET_FILEPATH = "..."
elif USER == {your_username}:
    JORDAN_DATASET_FILEPATH = {your_filepath}
```

- Run setup script to download Jordan dataset. Right now the setup script is just to download but if other setup needs to be done we can add it in there?
```
python -m src.setup
```

## 4. Normality testing
Set up proper multivariate normality testing (Mardia, Henze-Zirkler, Royston tests). To do this, install R integration:

```bash
# On hairesmobile (this should be sufficient):
conda install -c conda-forge gsl nlopt r-nloptr r-energy r-lme4 r-car r-mvn r-base

# Install rpy2 for Python-R bridge:
# might need to run: export RPY2_CFFI_MODE=ABI
pip install rpy2
```

<!-- I tried doing the installation via this method and it doesn't work for me: install the MVN package in R:
```r
# Start R
R

# Inside R console:
install.packages("MVN")
quit()
``` -->


**Note**: Without R/MVN, the code will fall back to univariate Shapiro-Wilk tests, which work but are less rigorous for multivariate data.

