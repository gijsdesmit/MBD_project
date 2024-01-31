from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_replace, isnan, when, count, col
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans, LDA, BisectingKMeans, GaussianMixture

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print('Imports done')

df = spark.read.csv('file:/home/s2559811/tsne_results.csv', header=True) # contains url, Dimension 1, Dimension 2
print('Loaded data')
df = df.sample(False, 0.5, seed=1) #todo: remove this

df = df.withColumn('Dimension 1', col('Dimension 1').cast('float'))
df = df.withColumn('Dimension 2', col('Dimension 2').cast('float'))

vecAssembler = VectorAssembler(inputCols=["Dimension 1", "Dimension 2"], outputCol="features")
df = vecAssembler.transform(df)

evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')

from sklearn.cluster import DBSCAN


# Convert Spark DataFrame to Pandas DataFrame
db_df = df.select("url","Dimension 1", "Dimension 2").toPandas()

# Apply DBSCAN clustering
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: The number of samples in a neighborhood for a point to be considered as a core point
#best big clustering: 3.5, 15

dbscan = DBSCAN(eps=3.2, min_samples=12) #3.2, 15 is kinda nice
db_df['prediction'] = dbscan.fit_predict(db_df[['Dimension 1', 'Dimension 2']])

# Display the first few rows of the DataFrame with cluster labels
print(db_df.head())

# Optional: Save the DataFrame with cluster labels to a new CSV file
db_df.to_csv('dbs_results.csv', index=False)

#### HDBSCAN

from hdbscan import HDBSCAN

# Convert Spark DataFrame to Pandas DataFrame
db_df = df.select("url","Dimension 1", "Dimension 2").toPandas()

# Apply HDBSCAN clustering
# min_cluster_size: The minimum number of samples in a neighborhood for a point to be considered as a core point

# BEST: 20, 10
hdbscan = HDBSCAN(min_cluster_size=20, min_samples=10, alpha=0.4)
db_df['prediction'] = hdbscan.fit_predict(db_df[['Dimension 1', 'Dimension 2']])

# Display the first few rows of the DataFrame with cluster labels
print(db_df.head())

# Optional: Save the DataFrame with cluster labels to a new CSV file
db_df.to_csv('hdbs_results.csv', index=False)

#########
#
# k = 200
# gmm = GaussianMixture().setK(k).setSeed(1).setFeaturesCol("features").setMaxIter(100)
# model = gmm.fit(df)
# output = model.transform(df)
# silhouette = evaluator.evaluate(output)
# print("Silhouette score = " + str(silhouette))
# output.show()
#
# # save output to csv
# output.drop('probability').drop('features').toPandas().to_csv('gmm_results.csv', index=False)
#
# ########
#
# km = KMeans(k=k, seed=1, featuresCol="features")
# model = km.fit(df)
# output = model.transform(df)
# silhouette = evaluator.evaluate(output)
# print("Silhouette score = " + str(silhouette))
# output.show()

###########

