# IIEG Employment Forecast

Simple regression techniques to perform employment forecast.

@author: Rodrigo Hernández Mota


## Usage

1. Create a .env file. 
    * `cp conf/.env.example conf/.env `

2. Create a virtualenv and install dependencies (WARNING: TODO).
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

## Model Parameters

Model parameters can be changed by editing the file: `model_setup.json`.

The `main.py` script will read this file as a python dictionary and
apply this configuration to the selected model. 

The file is structured as following: 
```
{
  "regression": {
    "random-forest": {
      "n-estimators": 50,
      "n-jobs": -1
    },
    "boosted-trees": {
      "max-depth": 10
    },
    "mlp": {
      "hidden-layers": "(100, 50, 20, 5)",
      "activation-function": "relu",
      "max-iter": 1000
    }
  }
}
```

Note that the hidden layer of the multi-layer perceptron are 
indicated as a python tuple between "". The main python script 
uses the function `eval()` on this parameter to pass the 
configuration to the scikit-learn library.


## Variable description

The raw dataset used in this repo can be found at `data/raw_data/insured_employment.pickle`.
This data is downloaded when running `setup.py`.

The data contains the following variables:
* **Economic Division (economic_division)** [categorical]
    * Agricultura, ganadería, silvicultura, pesca y caza
    * Comercio
    * Industria de la construcción
    * Industria eléctrica, captación y suministro de agua potable
    * Industrias extractivas
    * Servicios
    * Transportes y comunicaciones 
* **Gender (gender)** [categorical]
    * Hombres
    * Mujeres
* **Age Range (age_range)** [categorical]
    * De 15 a 19 años.
    * De 20 a 24 años.
    * De 25 a 29 años.
    * De 30 a 34 años.
    * De 35 a 39 años.
    * De 40 a 44 años.
    * De 45 a 49 años.
    * De 50 a 54 años.
    * De 55 a 59 años.
    * De 60 a 64 años.
    * De 65 a 69 años.
    * De 70 a 74 años.
    * De 75 ó más años
    * Menor a 15 años.
* **Year (year)** [numerical]
    * From 2006 to 2017.
* **Month (month)** [numerical]
    * From 1 to 12.
* **Number of insured employees (value)** [numerical]
    * From 1 to 46728.
   
See file `data-exploration.ipynb` for a more detailed analysis.

## Data Processing

The main objective of this project is to forecast the number of
insured employees to a given segregation level. Therefore, 
the response variable is the **number of insured employees** 
(shown as _value_ in the datasets) per year and month given 
the economic division, gender and age range.

The nature of this problem is of regression considering timeseries
data. 


    

## TODO

* Fix requirements for matplotlib version and dependencies.
