#!/usr/bin/env python
# coding: utf-8

# # Required Libraries

# In[209]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[205]:


input = pd.read_csv('playstore-analysis (2) (1).csv')
input


# In[68]:


input.describe()


# In[69]:


input.info()


# # 1. Data clean up – Missing value treatment

#     A . Drop records where rating is missing since rating is our target/study variable

# In[70]:


print(f"{input.isnull().sum()} \n are the missing values")


# In[94]:


data_a = input.dropna(subset = ["Rating"])


# In[95]:


print(f"{data_a.isnull().sum()} \nare the missing values")


#     B.Check the null values for the Android Ver column.

# In[96]:


print(f"Andriod ver column null values are:\n{input['Android Ver'].isnull().sum()}")


# In[97]:


print(data_a[data_a.isna().any(axis=1)])


#     Drop the 3rd record i.e. record for “Life Made WIFI …”

# In[113]:


data_b = data_a.drop(10472)


# In[108]:


#proof
proof = data_b[data_b["App"] == 'Life Made WI-Fi Touchscreen Photo Frame']
print(proof)


#     Replace remaining missing values with the mode

# In[143]:


import statistics
Android_list = data_b['Android Ver'].tolist()
Android_list
# Assuming the data
m = [1.5,4.0,4.0,4.2]
#print(statistics.mode(m))
data_new = data_b.fillna(statistics.mode(m))
print (data_new.loc[4453])
print('------------------------------------------------------')
print(print (data_new.loc[4490]))


# In[144]:


print(data_new.isnull().sum())


#     C. Current Ver – replace with most common value

#      Hence Their is "No" NA (Null Values) in the data , No need of replacing.

# # 2. Data Clean up – Correcting the Data types

#     A.Which all variables need to be brought to numeric types?

# In[158]:


data_new['Reviews'] = data_new['Reviews'].astype(int)
data_new['Size'] = data_new['Size'].astype(int)
data_new['Last Updated'] = data_new['Last Updated'].astype('datetime64[ns]')


# In[154]:


data_new.info() # Examining after conversion...


# In[165]:


print(data_new.dtypes) # Hence they are converted.


#     B.Price variable – remove $ sign and convert to float

# In[167]:


data_new['Price'] = data_new['Price'].str.replace('$','')
data_new.head(2)


# In[169]:


data_new['Price'] = data_new['Price'].astype(float)
data_new.info()


#     c. Installs – remove ‘,’ and ‘+’ sign, convert to integer

# In[171]:


data_new['Installs'] = data_new['Installs'].str.replace(',','')


# In[172]:


data_new['Installs'] = data_new['Installs'].str.replace('+','')


#     D. Convert all other identified columns to numeric

# In[174]:


data_new['Installs'] = data_new['Installs'].astype(int)


# In[175]:


data_new.head(2)


# In[176]:


data_new.info() #Hence 'Installs' have converted into int


# # 3. Sanity checks – Check for the following and handle accordingly

#     A. Avg. rating should be between 1 and 5, as only these values are allowed on the play store.

# In[192]:


pd.unique(data_new['Rating'].values) # Hence all the records are between 1 & 5.


#     B. Reviews should not be more than installs as only those who installed can review the app

# In[201]:


df = data_new[data_new.Reviews <= data_new.Installs]
df.head(2)


# In[203]:


df.shape


# # 4. Identify and Handle Outliers –

#     A. Price column

# In[222]:


#i. Make Suitable plot to identify outliers in price
plt.subplots(figsize=(12,10))
sns.boxplot(df.Price)

plt.xlabel('Unit Price')
plt.title('Unit Price')
plt.grid()
plt.show()


#     ii. Do you expect apps on the play store to cost $200? Check out these cases

# In[226]:


check = data_new[data_new['Price'] == 200]
print(check)

# Before data Cleaning..
#result = input[input['Price'] == '$200.00']
#print(result)


#     iii. After dropping the useless records, make the suitable plot again to identify outliers

# In[237]:


required_df = df[df['Price'] != 0]
plt.subplots(figsize=(20,10))
sns.boxplot(required_df.Price)

plt.xlabel('Unit Price')
plt.title('Unit Price')
plt.grid()
plt.show()


#     iv. Limit data to records with price < $30

# In[242]:


Pricelimit = df[df['Price'] < 30] 
(Pricelimit.head(5))


# In[244]:


df.shape


# B. Reviews column

#     i. Make suitable plot

# In[262]:


task = df.groupby('Category')['Reviews'].sum().sort_values()
plt.subplots(figsize=(20,10))
task.plot(kind='barh',fontsize=16)
print(task.sort_values(ascending = False))
plt.show()


#     i. Limit data to apps with < 1 Million reviews

# In[264]:


limit_data_app = df[df['Reviews'] < 1000000]
limit_data_app.head(2)


# C. Installs

#     i. What is the 95th percentile of the installs?

# In[266]:


Installs_percentage = df.Installs.quantile(0.95)
print(Installs_percentage)


# ii. Drop records having a value more than the 95th percentile

# In[268]:


check_for_records = df.Installs.quantile() > Installs_percentage
print(check_for_records) # No Records more than 95th percentile


# # Data analysis to answer Business questions

# 5. What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower 
# values?

# In[271]:


sns.distplot(df['Rating'])  # Left Skewed(Lower Values)
plt.show()


# b. What is the implication of this on your analysis?

# In[272]:


df['Content Rating'].value_counts()


# In[273]:


Adult_rating = df[df['Content Rating'] == 'Adults only 18+'].index.to_list()
unrated =df[df['Content Rating'] == 'Unrated'].index.to_list()
df.drop(Adult_rating, inplace = True)
df.drop(unrated, inplace = True)
df['Content Rating'].value_counts()


# In[ ]:


# Pie Diagram
import plotly.graph_objects as go

fig = go.Figure(go.Pie(
    name = "",
    values = [7414,1083,461,397],
    labels = ['Everyone','Teen','Mature 17+','Everyone 10+'],
))
fig.show()


# 7. Effect of size on rating

# A.Make a joinplot to understand the effect of size on rating

# In[286]:


sns.jointplot(x=df['Size'],y=df['Rating'],data=df,kind='hex')
plt.show()


# b. Do you see any patterns? c. How do you explain the pattern?

# 1.Effect of price on rating
# 
# a. Make a jointplot (with regression line)

# In[287]:


sns.jointplot(x ="Rating" , y = "Price" ,data = df)
plt.show()


# b. What pattern do you see?
# 
# c. How do you explain the pattern?
# 
# d. Replot the data, this time with only records with price > 0

# In[289]:


Price_greaterthan_zero = df[df['Price'] > 0]
sns.jointplot(x ="Price" , y = "Rating" ,data = Price_greaterthan_zero, kind = "reg" )
plt.show()


# In[290]:


sns.lmplot(x='Price', y='Rating', hue ='Content Rating', data=df)
plt.show()


# 1.Look at all the numeric interactions together – a. Make a pairplort with the colulmns - 'Reviews', 'Size', 'Rating', 'Price'

# In[292]:


sns.pairplot(df,vars=['Rating','Size', 'Reviews', 'Price'])
plt.show()


# 1.Rating vs. content rating
# 
# a. Make a bar plot displaying the rating for each content rating

# In[293]:


a = df['Rating'].groupby(df['Content Rating']).median().plot(kind = 'bar')
a.set(xlabel ='Rating of content', ylabel = 'Average of Ratings')
plt.show()


# b. Which metric would you use? Mean? Median? Some other quantile?
# 
# c. Choose the right metric and plot

# In[299]:


df.groupby(['Content Rating'])['Rating'].count().plot.bar(color="g")
plt.ylabel('Rating')
plt.show()


# 1.Content rating vs. size vs. rating – 3 variables at a time
# 
# a. Create 5 buckets (20% records in each) based on Size

# In[300]:


#Checking skewness
sns.distplot(df["Size"], bins=5)
plt.show()


# In[301]:


bins=[0, 4600, 12000, 21516, 32000, 100000]
df['Size_Buckets'] = pd.cut(df['Size'], bins, labels=['VERY LOW','LOW','MED','HIGH','VERY HIGH'])
pd.pivot_table(df, values='Rating', index='Size_Buckets', columns='Content Rating')


# b. By Content Rating vs. Size buckets, get the rating (20th percentile) for each combination

# In[302]:


df.Size.quantile([0.2, 0.4,0.6,0.8])


# In[303]:


df.Rating.quantile([0.2, 0.4,0.6,0.8])


# c. Make a heatmap of this
# 
# i. Annotated
# 
# ii. Greens color map

# In[305]:


Size_Buckets =pd.pivot_table(df, values='Rating', index='Size_Buckets', columns='Content Rating', 
                     aggfunc=lambda x:np.quantile(x,0.2))
Size_Buckets


# In[306]:


sns.heatmap(Size_Buckets, annot = True)
plt.show()


# In[307]:


sns.heatmap(Size_Buckets, annot=True, cmap='Greens')
plt.show()

