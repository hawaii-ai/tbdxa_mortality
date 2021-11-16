# TBDXA Mortality Prediction 
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
### Github repository containing all relevant code for Nature Communications in Medicine submission: DEEP LEARNING IDENTIFIES BODY COMPOSITION CHANGES OVER TIME IN TOTAL-BODY DXA IMAGING TO PREDICT ALL-CAUSE MORTALITY

## Installation and system requirements:
- Tested on CentOS Linux 7 (Core)
- Python version: 3.8.10
- To install dependencies, run:
```
python setup.py install
```
- Installation time < 5 minutes

## Demo
- Demo script are provided in the notebooks folder
- A demo dataset is provided purely to validate model functionality, the dataset is not representative of the data used to train/ evaluate the models referenced in the manuscript
- To validate code functionality, run sample code in notebook corresponding to desired functionality (e.g. model_loading.ipynb for an example of how to load and test pretrained models)
- Execution time varies depending on hardware, but training and evaluation on the demo dataset should generally take no more than five minutes
- Code to recreate figures and statistical analyses from the manuscript is provided in modules/utils/analysis_scripts.py, more extensive example notebooks for these will be provided in the future


This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. 

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

<img src="./images/tbdxa_github_qr.svg" width="85">

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
