# OpenClassrooms Machine Learning Engineer Path

## Project 4 : Segment customers of an e-commerce site

### Project description :

This project is a clustering algorithm which segment customers from Olist which is a Brazilian E-Commerce platform.

**Problematic :**

_You are a consultant for Olist, a sales solution on online marketplaces._

_Olist wants you to provide their e-commerce teams with **customer segmentation** that they can use on a daily basis for their communication campaigns._

_Your goal is to **understand the different types of users** through their behavior and their personal data._

_You will need to **provide the marketing team with an actionable description of your segmentation** and its underlying logic for optimal use, as well as a **maintenance contract proposal** based on an analysis of the stability of the segments over time._


**Methodology :**

Here we measure the quality of customer segmentation by comparing two approaches :
- a **digital marketing approach** based on **RFM segmentation** (recency, frequency, monetary value)
- a **machine learning approach** by selecting the **clustering algorithm** that best corresponds to this type of problem (here **k-means**)

The selected customer segmentation corresponds to the one that provides the most granularity (the one that is the most actionable) for the sales teams.

It was also necessary to assess the frequency with which the segmentation must be updated, in order to be able to carry out a maintenance contract estimate.


**Evaluation results :**


The segmentation selected corresponds to that resulting from the machine learning approach (k-means) which gives 11 clusters (against 7 for the RFM segmentation) :

| Segment | Order frequency | Season | Month period | Average price | Average freight | Average payment | Lifetime (months) |
| ------  | --------------- | ------ | ------------- | ------------- | --------------- | --------------- |  --------------- |
|**1 - Best Customers** | high | spring | beginning | very high | very high | very high | 19 |
|**2 - Cheap New Customers** | low | spring | mid-month | very low | very low | very low | 6 |
|**3 - Promising Customers** | high | winter | beginning | low | medium | low | 19 |
|**4 - Highest Paying Customers**| low | spring | beginning | very high | very high | very high | 19 |
|**5 - Promising Customers Lost** | high | summer | beginning | low | low | low | 16 |
|**6 - Cheap Customers** | low | winter | mid-month | low | high | low | 12 |
|**7 - Impulse Customers Lost** | low | summer | mid-month | high | high | high | 10 |
|**8 - Best New Customers** | high | spring | beginning | high | high | high | 18 |
|**9 - Loyal Customers** | very high | spring | beginning | high | high | high | 19 |
|**10 - Impulse New Customers** | low | spring | mid-month | high | high | high | 6 |
|**11 - Cheap Customers Lost** | low | summer | mid-month | very low | very low | very low | 11 |


**Main data :**

For this mission, Olist has provided you with an [anonymized database](https://www.kaggle.com/olistbr/brazilian-ecommerce) with information on order history, products purchased, satisfaction reviews, and customer location since January 2017.

### Project architecture :

- **_notebooks_html_** folder which contains jupyter notebooks in HTML format.
- **_notebooks_ipynb_** folder which contains jupyter notebooks in .ipynb format.
- **_src_** folder which contains several personal packages and modules coded in python
- **_utils_** folder which contains the main module of this project (_olist.py_)

### Description of notebooks :

- **[POLIST_01_notebook_exploratory_data_analysis.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Customer-Segmentation/notebooks_ipynb/POLIST_01_notebook_exploratory_data_analysis.ipynb)** which covers the cleaning & analysis steps (EDA)
- **[POLIST_02_notebook_modelization_sampled_frequency.ipynb](https://github.com/4D1L-PY/Portfolio/blob/main/OC-MLE/Customer-Segmentation/notebooks_ipynb/POLIST_02_notebook_modelization_sampled_frequency.ipynb)** which corresponds to the main modeling work where the purchase frequency variable has been normalized.
