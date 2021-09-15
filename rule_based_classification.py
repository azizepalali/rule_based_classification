##################################################
# Let's make it more meaningful together                  
##################################################

# İnstalling required libraries
import numpy as np 
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', True)



# Get the datasets of persona
df = pd.read_csv("hafta_2/persona.csv")
df.head()



######### Understanding the data sets   ######

df.ndim             # ndim: Number of dimensions.
df.shape            # shape: shape of datasets
df.size             # size: numbers of elements
df.head()           # head: first five values of datasets
df.columns          # columns: name of columns
df.info             # info: get help information for a function, class, or module.
df.describe().T
df.isnull().values.any()  # answer of null value
df.isnull().sum()         # total of null value

def check_df(dataframe):
    print(f"""
        ##################### Shape #####################\n\n\t{dataframe.shape}\n\n
        ##################### Types #####################\n\n{dataframe.dtypes}\n\n
        ##################### Head #####################\n\n{dataframe.head(3)}\n\n
        ##################### NA #####################\n\n{dataframe.isnull().sum()}\n\n
        ##################### Quantiles #####################\n\n{dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}\n\n""")
check_df(df)


######## Selection of Categorical and Numerical Variables ######## 

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

######## General Exploration for Categorical Data ######## 

def cat_summary(dataframe, plot=False):
    # cat_cols = grab_col_names(dataframe)["Categorical_Data"]
    for col_name in cat_cols:
        print("############## Unique Observations of Categorical Data ###############")
        print("The unique number of " + col_name + ": " + str(dataframe[col_name].nunique()))

        print("############## Frequency of Categorical Data ########################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:  # plot == True (Default)
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()
            
 cat_summary(df, plot=True)           
            
            
######## General Exploration for Numerical Data ######## 

def num_summary(dataframe, plot=False):
    numerical_col = ['PRICE', 'AGE']  ##or grab_col_names(dataframe)["Numerical_Data"]
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in numerical_col:
        print("########## Summary Statistics of " + col_name + " ############")
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name)
            plt.xlabel(col_name)
            plt.title("The distribution of " + col_name)
            plt.grid(True)
            plt.show(block=True)


num_summary(df, plot=True)   


######## Analysis of the data set ######## 
    
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

# mean of sales by source,country,sex and age as descending order

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).sort_values(("PRICE"),
                                                                                        ascending=False)

agg_df.head() 


# Solving the index problem
agg_df = agg_df.reset_index()



######## Define New level-based Customers ######## 
# But, firstly we need to convert age variable to categorical data.

bins = [agg_df["AGE"].min(), 18, 23, 35, 45, agg_df["AGE"].max()]
labels = [str(agg_df["AGE"].min())+'_18', '19_23', '24_35', '36_45', '46_'+ str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins , labels=labels)
agg_df.groupby("AGE_CAT").agg({"AGE": ["min", "max", "count"]})


# For creating personas, we group all the features in the dataset:

agg_df = agg_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE_CAT", "AGE"])[["PRICE"]].sum().reset_index()
agg_df["CUSTOMERS_LEVEL_BASED"] = pd.DataFrame(["_".join(row).upper() for row in agg_df.values[:,0:4]])
agg_df.head()

# Calculating average amount of personas:

agg_df.groupby('CUSTOMERS_LEVEL_BASED').agg({"PRICE": "mean"})

# group by of personas:

agg_df = agg_df.groupby('CUSTOMERS_LEVEL_BASED').agg({"PRICE": "mean"}).sort_values(("PRICE"),ascending=False).reset_index()
agg_df.head()


######## Creating Segments based on Personas   ######## 

segment_labels = ["D","C","B","A"]
agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"], 4, labels=segment_labels)

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}).sort_values(("SEGMENT"),
                                                                             ascending=False).reset_index()

# Demonstrating segments as bars on a chart, where the length of each bar varies based on the value of the customer profile
   
plot = sns.barplot(x="SEGMENT", y="PRICE", data=agg_df)

for bar in plot.patches:
    plot.annotate(format(bar.get_height(), '.2f'),
             (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                      textcoords='offset points')
    
    
#########   Prediction     ######## 

def ruled_based_classification(dataframe):

    def AGE_CAT(age):
        if age <= 18:
            AGE_CAT = "15_18"
            return AGE_CAT
        elif (age > 18 and age <= 23):
            AGE_CAT = "19_23"
            return AGE_CAT
        elif (age > 23 and age <= 35):
            AGE_CAT = "24_35"
            return AGE_CAT
        elif (age > 35 and age <= 45):
            AGE_CAT = "36_45"
            return AGE_CAT
        elif (age > 45 and age <= 66):
            AGE_CAT = "46_66"
            return AGE_CAT

    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANDROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_SEG

    print(new_user)
    print("Segment:" + agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "SEGMENT"].values[0])
    print("Price:" + str(agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))

    return new_user


ruled_based_classification(df)    
