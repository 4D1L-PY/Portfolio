# OpenClassrooms Machine Learning Engineer Path

## Project 3: Anticipate the electricity consumption needs of buildings

### Project description :

This project is a regression algorithm which predict electricity consumption and gas emissions for the city of Seattle.

**Problematic :**

_You work for the city of Seattle. To achieve its goal of being a neutral city in carbon emissions by 2050, your team is paying close attention to the emissions of buildings not intended for housing._

_**Consumption records** were taken by your agents in **2015** and **2016**. However, these records are expensive to obtain, and from those already taken, you want to attempt to **predict the CO2 emissions and total energy consumption of buildings**. for which they have not yet been measured._

_Your prediction will be based on the **declarative data of the commercial exploitation permit** (size and use of buildings, mention of recent work, date of construction, etc.)_

_You are also looking to gauge the value of the **"ENERGY STAR Score"** for predicting emissions, which is tedious to calculate with the approach your team currently uses._

**Methodology :**

First we have to build a panel of models to train.

Then we evaluate the importance of the features for each of the different models of this panel, in order to select those which contribute the most to the performance of the models.

Particular attention was paid to the feature of the **ENERGY STAR Score** (which is expensive to acquire) by evaluating its contribution to model performance (including and excluding it).

Since we have 2 targets, we have to select the best model for each of these targets by following the tradeoff between :
- model performances (**RMSE**, R2)
- cost of acquisition of features (total features)
- complexity of models
- potential for improvement (with more data from 'training)

In this case it turns out that the most efficient model is **XGBoost** and it is **common to both targets**.

**Evaluation results :**
- Energy consumption target (_SiteEnergyUse(kBtu)_) :

|  Model   | RMSE | R2 |  Selected features |  Learning potential  |
| -------  | ---- | -- |  ----------------- |  ------------------- |
|    ElasticNet     | 0.56 |  0.75 |  31 | No |
|    RandomForestRegressor     | 0.48 |  0.82 |  12 | Yes |
|    **XGBRegressor**     | **0.45** |  **0.84** |  **12** | **Yes** |
|    SVR     | 0.59 |  0.73 |  12 | No |
|    MLPRegressor     | 0.59 |  0.72 |  12 | No |

- Gas emissions target (_TotalGHGEmissions_) :

|  Model   | RMSE | R2 |  Selected features |  Learning potential  |
| -------  | ---- | -- |  ----------------- |  ------------------- |
|    ElasticNet     | 0.57 |  0.74 |  36 | No |
|    RandomForestRegressor     | 0.47 |  0.82 |  18 | Yes |
|    **XGBRegressor**     | **0.46** |  **0.84** |  **18** | **Yes** |
|    SVR     | 0.57 |  0.74 |  18 | No |
|    MLPRegressor     | 0.58 |  0.74 |  18 | Yes |


**N.B :**
- **selected features** refers to **main needed variables** of the model
- **learning potential** measures the **potential for improvement of the model if it had access to more data**


**Main data :**

The consumption data can be downloaded at the following link : [SEA Building Energy Benchmarking](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv)

### Project architecture :

- **_notebooks_html_** folder which contains jupyter notebooks in HTML format.
- **_notebooks_ipynb_** folder which contains jupyter notebooks in .ipynb format.
- **_src_** folder which contains several personal packages and modules coded in python.

### Description of notebooks :

- **In total there are 5 jupyter notebooks** which break down the work done by distinguishing the steps of **cleaning**, **analysis**, **modeling** of data and **evaluation** of models :
    - **[Pelec_01_notebook_exploratory_data_analysis.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Electricity-Consumption/notebooks_ipynb/Pelec_01_notebook_exploratory_data_analysis.ipynb)** which covers the cleaning & analysis steps (EDA) by **excluding the variable ENERGYSTARScore** including:
        - **the acquisition is expensive**
        - **the relevance for the modelization is not major** (performance difference of about **10%** concerning the mean square deviation (**RMSE**) or even with the coefficient of determination (**R2**))
    - **[Pelec_02_notebook_final_model_selection.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Electricity-Consumption/notebooks_ipynb/Pelec_02_notebook_final_model_selection.ipynb)** which takes up the step of evaluating the models according to the inclusions and exclusions of the variables mentioned above. **(N.B: this notebook presents the final models that have been selected for each target)**
    - **[Pelec_03_notebook_modelization_exclude_ENERGYSTARScore.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Electricity-Consumption/notebooks_ipynb/Pelec_03_notebook_modelization_exclude_ENERGYSTARScore.ipynb)** which resumes the data modeling step (**excluding the variable ENERGYSTARScore**)
    - **[Pelec_04_notebook_modelization_exclude_ENERGYSTARScore_and_main_energy_vars.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Electricity-Consumption/notebooks_ipynb/Pelec_04_notebook_modelization_exclude_ENERGYSTARScore_and_main_energy_vars.ipynb)** which resumes the data modeling step (**excluding the variables ENERGYSTARScore, Main_energy_electricity, Main_energy_steam**)
    - **[Pelec_05_notebook_modelization_include_ENERGYSTARScore.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Electricity-Consumption/notebooks_ipynb/Pelec_05_notebook_modelization_include_ENERGYSTARScore.ipynb)** which resumes the data modeling step (**including the variable ENERGYSTARScore**)
