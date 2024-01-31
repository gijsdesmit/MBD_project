from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hash, monotonically_increasing_id, udf, explode
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import split, regexp_replace, isnan, when, count, col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print('Imports done')

# df = spark.read.json('/data/doina/WebInsight/*/*.gz') # all data
df = spark.read.json('/data/doina/WebInsight/2020-09-07/*.gz')  # one week of data

print('Loaded data')

# Get semanticVector and url
df = df.select(df['url'], df['fetch.semanticVector'].alias('semanticVector'))
df = df.filter(col('semanticVector').startswith("["))
df = df.withColumn('semanticVector', split(regexp_replace('semanticVector', r'(\[|\])', ''), ', '))
df = df.withColumn('semanticVector', df['semanticVector'].cast(ArrayType(FloatType())))
#use this df for tsne

# Convert list of floats to dense vector
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn('semanticVector', list_to_vector_udf(df['semanticVector']))


def run_pca(k):
    pca = PCA(k=k, inputCol="semanticVector", outputCol="pcaFeatures")
    model = pca.fit(df)
    result = model.transform(df).select("pcaFeatures")
    result.show()
    return result


def find_optimal_k():
    # Fit the PCA model to your data with k equal to the total number of features
    pca = PCA(k=192, inputCol="semanticVector", outputCol="pcaFeatures")
    model = pca.fit(df)
    explainedVariance = model.explainedVariance
    cumulativeExplainedVariance = np.cumsum(explainedVariance)
    print(cumulativeExplainedVariance)
    # Find the smallest k such that the cumulative explained variance is greater than 0.95
    k = np.argmax(cumulativeExplainedVariance > 0.95) + 1
    print(f'The optimal number of principal components is {k}')
    return k


def run_tsne():
    pd_df = df.toPandas()
    tsne = TSNE(n_components=2, random_state =0)
    sv = np.array(pd_df['semanticVector'].tolist())
    tsne_results = tsne.fit_transform(sv)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dimension 1', 'Dimension 2'])
    print(tsne_df.head())
    #save tsne_df to csv
    tsne_df.to_csv('tsne_results.csv', index=False)


k = find_optimal_k()
result = run_pca(k)

#try taking a random sample (that completes within an hour) as representative for TSNE
