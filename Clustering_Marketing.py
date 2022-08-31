import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.decomposition import PCA

#  MORE INFO ABOUT THE DATASET - https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

base = pd.read_csv('Marketing_data.csv')
df = pd.read_csv('Marketing_data.csv')

# PRINT 1

#  todo EXPLORATION ------------------------

'''For a better overview I`m using describe and info so we can understand the basics of this dataset'''

print(base.info())
print(base.describe())

# PRINT 1B

'''We can check for frauds filtering the highest debits and cash payments in the dataset'''

high_deb = df[df['ONEOFF_PURCHASES'] > 20000]
high_cash = df[df['CASH_ADVANCE'] > 20000]

print(high_deb)
print(high_cash)

# PRINT 2

'''Now we need to check Nan data, and duplicated data'''

print(df.isnull().sum())
print(df.duplicated().sum())

# PRINT 3

'''Since we have some NaN data, let's fill it with the mean of the same columns so we don't have to delete
info from the dataset, some infos in other parts of the dataset might be important, so we I opted to not drop
this data, and we're also dropping the customer id'''

df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())
df.drop(columns='CUST_ID', inplace=True)

'''Let's also apply this to the original dataset'''

base['MINIMUM_PAYMENTS'] = base['MINIMUM_PAYMENTS'].fillna(base['MINIMUM_PAYMENTS'].mean())
base['CREDIT_LIMIT'] = base['CREDIT_LIMIT'].fillna(base['CREDIT_LIMIT'].mean())

'''Now let's take a deeper look at the data and its distribution'''

plt.figure(figsize=(10, 50))
for i in range(len(df.columns)):
    plt.subplot(17, 1, i + 1)
    sns.histplot(df[df.columns[i]], kde=True)
    plt.title(df.columns[i])
plt.tight_layout()
plt.show()

# PRINT 4

'''And the correlation between the variables'''

correlations = df.corr()
f, ax = plt.subplots(figsize=(25, 18))
sns.heatmap(correlations, annot=True)
plt.show()

'''We can see that we have some interesting areas for correlation, for example, the amount of credit a person
have, do not makes difference if he's paying or not, there's many insights we can explore if we need'''

# PRINT 5

#  todo NORMALIZATION ------------------------

'''Simple data normalization since we don't have any categorical information'''

standardscaler = StandardScaler()
df = standardscaler.fit_transform(df)

# PRINT 6

#  todo DIMENSIONALITY REDUCTION ------------------------

'''At this point to achieve a good result, we'll try a deep learning approach with autoencoders to reduce the
dimensionality of the dataset, with a Dense network and activation with relu which mean that negative
numbers will be converted to 0 and positives will be passed ahead while it adjust the weights'''

input = Input(shape=(17,))
X = Dense(500, activation='relu')(input)
X = Dense(2000, activation='relu')(X)

encoded = Dense(10, activation='relu')(X)
X = Dense(2000, activation='relu')(encoded)
X = Dense(500, activation='relu')(X)

decoded = Dense(17)(X)

autoencoder = Model(input, decoded)
encoder = Model(input, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(df, df, epochs=50)

'''Then we can check the results and dimensionality of the dataset'''

print(df.shape)
compact_df = encoder.predict(df)
print(compact_df.shape)

# PRINT 7

#  todo ELBOW TEST ------------------------

'''I`m using the elbow method to check the WCSS, so we can define how many clusters would be the optimal,
basically we are using the centroid position with euclidian distance to check how many clusters (groups)
would be the optimal'''

wcss1 = []
ranged = range(1, 20)
for i in ranged:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(compact_df)
    wcss1.append(kmeans.inertia_)

plt.plot(wcss1, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

'''Which we can observe and conclude that the optimal number of cluster would be around 4 (without the dimensionality
we would be working with around 8 clusters, so we had an improvement'''

# PRINT 8

#  todo MODEL CREATION ------------------------

'''Let's create our model'''

kmeans = KMeans(n_clusters=4)
kmeans.fit(compact_df)

labels = kmeans.labels_
print(labels.shape)

flabel = pd.DataFrame({'cluster': labels})
df_clustered = pd.concat([base, pd.DataFrame({'cluster': labels})], axis=1)

'''Now we have a dataset with the cluster each of the client belongs, which are correlated by many factors'''

# PRINT 9

#  todo PCA ------------------------

'''Since we're using autoencoder, we're not using PCA for this dataset, but this could be a good option'''

pca = PCA(n_components=2)
prin_comp = pca.fit_transform(compact_df)
pca_df = pd.DataFrame(data=prin_comp, columns=['pca1', 'pca2'])

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df, palette=['red', 'green', 'blue', 'pink'])
plt.show()

'''PCA in this case would create a 2D view of some 3D data, this way we can use it for dimensionality problems
and for data visualization'''

# PRINT 10

#  todo DATA ANALYSIS ------------------------

'''Let's split the dataset for further information about the clusters our model created'''

df_clustered = df_clustered.sort_values(by='cluster')

group0 = df_clustered[df_clustered['cluster'] == 0]
group1 = df_clustered[df_clustered['cluster'] == 1]
group2 = df_clustered[df_clustered['cluster'] == 2]
group3 = df_clustered[df_clustered['cluster'] == 3]

'''We can check some basic data and look for insights with a basic visualization of all the clusters created'''

print(group0.describe(), group1.describe(), group2.describe(), group3.describe())

# PRINT 10b

'''Let's plot some figures to compare each parameter od the clusters, this way we can check the differences about
the clusters the autoencoder selected'''

#  todo COMPARISON PLOTS ------------------------

for i in df_clustered.columns:
    plt.figure(figsize=(25, 5))
    for j in range(4):
        plt.subplot(1, 4, j + 1)
        cluster = df_clustered[df_clustered['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title(f'{i} \nCluster {j}')
    plt.show()

# PRINT 11

'''Now we can compare group by group and have a better idea which one is grouped by how much they spend, or
how much they earn, if there's a correlation, we'll be able to see, such as group 0, that have the best
credit card limit, group 3 seems to be more economics, and so on...now we can export the data for deeper analysis'''

#  todo SAVING THE DATASET ------------------------

df_clustered.to_csv('cluster_customers.csv')
