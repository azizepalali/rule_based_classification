##################################################
#                     GÖREV 1
##################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Get the datasets of persona
df = pd.read_csv("hafta_2/persona.csv")
# Analyzing the data sets

df.ndim             # ndim: Number of dimensions.
df.shape            # shape: shape of datasets
df.size             # size: numbers of elements
df.head()           # head: first five values of datasets
df.columns          # columns: name of columns
df.info             # info: get help information for a function, class, or module.
df.describe().T
df.isnull().values.any()  # answer of null value
df.isnull().sum()         # total of null value

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Unique Value of source
df["SOURCE"].unique()

# Frequency of source
df["SOURCE"].value_counts()
df.groupby("SOURCE").count()

# Unique Value and count of price
df["PRICE"].unique()
df.groupby("PRICE").count()

# sales by country
df.groupby("COUNTRY")["PRICE"].count()

# total sales by country
df.groupby("COUNTRY")["PRICE"].sum()

# sales by source
df.groupby("SOURCE")["PRICE"].count()

# mean of sales by country
df.groupby("COUNTRY")["PRICE"].mean()

# mean of sales by source
df.groupby("SOURCE")["PRICE"].mean()

# mean of sales by source and country
df.groupby(["SOURCE", "COUNTRY"])["PRICE"].mean()

##################################################
#                    GÖREV 2
##################################################

# mean of sales by source,country,sex and age
df.groupby(["COUNTRY","SOURCE", "SEX", "AGE"]).agg({'PRICE':'mean'})


##################################################
#                    GÖREV 3
##################################################

# mean of sales by source,country,sex and age as descending order

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).sort_values(("PRICE"),
                                                                                        ascending=False)


##################################################
#                    GÖREV 4
##################################################
# reset index
agg_df = agg_df.reset_index()

agg_df["AGE"].min()

##################################################
#                   GÖREV 5
##################################################
# Let's define new level-based customers (personas) by using Country, Source, Age and Sex.
# But, firstly we need to convert age variable to categorical data.

bins = [agg_df["AGE"].min(), 18, 23, 35, 45, agg_df["AGE"].max()]
labels = [str(agg_df["AGE"].min())+'_18', '19_23', '24_35', '36_45', '46_'+ str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins , labels=labels)
agg_df.head()

##################################################
#                  GÖREV 6
##################################################
# For creating personas, we group all the features in the dataset:

agg_df = agg_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE_CAT", "AGE"])[["PRICE"]].sum().reset_index()
agg_df["CUSTOMERS_LEVEL_BASED"] = pd.DataFrame(["_".join(row).upper() for row in agg_df.values[:,0:4]])

# Calculating average amount of personas:

agg_df.groupby('CUSTOMERS_LEVEL_BASED')[['PRICE']].mean()
agg_df.sort_values("PRICE",ascending=False)
agg_df.head()

# group by of personas:
agg_df = agg_df.groupby('CUSTOMERS_LEVEL_BASED').agg({"PRICE": "mean"}).sort_values(("PRICE"),ascending=False)
agg_df = agg_df.reset_index()
##################################################
#                    GÖREV 7
##################################################
# Describe segment columns

segment_labels = ["D","C","B","A"]
agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"], 4, labels=segment_labels)

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}).sort_values(("SEGMENT"),
                                                                             ascending=False).reset_index()

# Describe segment columns
agg_df.groupby("SEGMENT").agg({"PRICE":"mean"}).describe().T


# Analysis of Segment of C
agg_df[agg_df["SEGMENT"] == "C"]["PRICE"].describe().T

##################################################
#GÖREV 8
##################################################
# Prediction of customer value

new_user= "CAN_IOS_FEMALE_36_45"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user ]


