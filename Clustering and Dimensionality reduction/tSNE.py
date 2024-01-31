from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import split, regexp_replace, when, count, col
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler

#spark = SparkSession.builder.getOrCreate()
spark = SparkSession.builder.config("spark.dynamicAllocation.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print('Imports done')

#df = spark.read.json('/data/doina/WebInsight/*/*.gz') # all data
df = spark.read.json('/data/doina/WebInsight/2020-09-07/*.gz') # one week of data
df_relurls = spark.read.parquet('/user/s2350726/relurls.parquet') # relevant urls to keep

print('Loaded data')

# Get semanticVector and url
df = df.select(df['url'], df['fetch.semanticVector'].alias('semanticVector'))
df = df.join(df_relurls, 'url', 'inner') # Remove all rows that are not in relurls
df = df.withColumn('semanticVector', split(regexp_replace('semanticVector', r'(\[|\])', ''), ', '))
df = df.withColumn('semanticVector', df['semanticVector'].cast(ArrayType(FloatType())))
#df = df.limit(90000) #TODO: remove this limit

#Randomly sample fraction of the rows
df = df.sample(False, 0.12, seed=1)

# Semantic vector has 192 elements in each vector, and we aim to reduce it to X elements using an autoencoder

#df_train = df.withColumn('label', df['semanticVector']) # Convert semanticVector to label for training
df_train = df.withColumnRenamed('semanticVector', 'features')
df_train_pd = df_train.toPandas()

######### Applying t-SNE

data = np.stack(df_train_pd['features'].values)

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Perform t-SNE
# best: perplexity=40, n_iter=2000
tsne = TSNE(n_components=2, random_state=0, perplexity=40, n_iter=2000)
tsne_results = tsne.fit_transform(data)

# Convert the result to a DataFrame
df_tsne = pd.DataFrame(data=tsne_results, columns=['Dimension 1', 'Dimension 2'])
df_tsne['url'] = df_train_pd['url'].values
df_tsne.to_csv('tsne_results.csv', index=False)

