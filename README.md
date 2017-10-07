# IIEG Employment Forecast

Simple regression techniques to perform employment forecast.

@author: Rodrigo Hernández Mota


## Usage

1. Create a .env file. 
    * `cp conf/.env.example conf/.env `

2. Create a virtualenv and install dependencies.
    * `virtualenv venv`
    * `source activate venv`
    * `pip install -r requirements.txt`
    
3. Boostrap the project.
    * `python setup.py`
    
4. [Optional] Modify the file `model_setup.json` to specify the parameters to use.
    * `nano model_setup.json`
    
5. Run the main python script.
    * `python main.py --model mlp --plot true`
    * Available models: `mlp`, `rf`.
    
Results are printed on the console and saved into the dir `logs`.

## Variable description

The raw dataset used in this repo can be found at `data/raw_data/insured_employment.pickle`.

The data contains the following variables:
* **Economic Division (economic_division)** categorical 
* **Gender (gender)** categorical
* **Age Range (age_range)** categorical
* **Year (year)** numerical
* **Month (month)** numerical
* **Number of insured employees (value)** numerical