from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hash, monotonically_increasing_id, udf, explode
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import split, regexp_replace, isnan, when, count, col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans, LDA, BisectingKMeans, GaussianMixture

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print('Imports done')

#df = spark.read.json('/data/doina/WebInsight/*/*.gz') # all data
df = spark.read.json('/data/doina/WebInsight/2020-09-07/*.gz') # one week of data
df_relurls = spark.read.parquet('/user/s2350726/relurls.parquet') # relevant urls to keep

print('Loaded data')

# Get semanticVector and url
df = df.select(df['url'], df['fetch.semanticVector'].alias('semanticVector'))
# Remove all rows that are not in relurls
df = df.join(df_relurls, 'url', 'inner')
# df = df.filter(col('semanticVector').startswith("[")) # Don't need this anymore because of the join
df = df.withColumn('semanticVector', split(regexp_replace('semanticVector', r'(\[|\])', ''), ', '))
df = df.withColumn('semanticVector', df['semanticVector'].cast(ArrayType(FloatType())))
#df = df.limit(90000) #TODO: remove this limit

# Convert list of floats to dense vector
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn('semanticVector', list_to_vector_udf(df['semanticVector']))

# Convert 'semanticVector' column to a feature vector
vecAssembler = VectorAssembler(inputCols=["semanticVector"], outputCol="features")
df_kmeans = vecAssembler.transform(df).drop('semanticVector')

# Use same evaluator for all clustering algorithms
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='cosine')

n = df_kmeans.count()
print(f'Number of rows in df_kmeans: {n}')
df_kmeans = df_kmeans.limit(n) # This is needed because of a bug?

# I don't think we need to scale the features because the values are already between -1 and 1

# List of k (#clusters) values to try
k_list = [2,50,100,200,350,500,750,1000]

def kmeans():
    print('Starting KMeans fit')
    wssse_values =[]

    for k in k_list:
        kmeans = KMeans(k=k, seed=1, featuresCol="features") # set maxIter?
        model = kmeans.fit(df_kmeans)
        output = model.transform(df_kmeans)
        score = evaluator.evaluate(output)
        print(f'k={k} -> sillhouette score={score}')
        wssse_values.append(score)

    # Some results:
    # k=2 -> sillhouette score=0.018
    # k=50 -> sillhouette score=0.074
    # k=100 -> sillhouette score=0.099
    # k=200 -> sillhouette score=0.13

    #TODO: use cluster centers to extract meaningful labels for clusters
    #centers = model.clusterCenters()


def run_lda():
    # We could not get LDA to run properly nor distributively (with completion), even on small data

    print('Starting LDA fit')

    scores = []
    # Trains a LDA model.
    for k in k_list:
        lda = LDA(k=k, maxIter=1, featuresCol="features")
        model = lda.fit(df_kmeans) #TODO: maybe set --min-executors, num-executots /--help

        # Describe topics.
        topics = model.describeTopics(3) #?????
        print("The topics described by their top-weighted terms:")
        topics.show(truncate=False)

        # Shows the result
        output = model.transform(df_kmeans)
        #output.show(truncate=False)
        score = evaluator.evaluate(output)
        print(f'k={k} -> sillhouette score={score}')
        scores.append(score)


def run_bkmeans(): #bisecting kmeans
    print('Starting Bisecting-KMeans fit')
    wssse_values =[]

    for k in k_list:
        kmeans = BisectingKMeans(k=k, seed=1, featuresCol="features", maxIter=30)
        model = kmeans.fit(df_kmeans)
        output = model.transform(df_kmeans)
        score = evaluator.evaluate(output)
        print(f'k={k} -> sillhouette score={score}')
        wssse_values.append(score)

def run_gmm():
    # We could not get GMM to run distributively, even on small data

    print('Starting GMM fit')
    wssse_values =[]

    for k in k_list:
        gmm = GaussianMixture().setK(2).setSeed(1).setFeaturesCol("features").setMaxIter(50)
        #gmm = GaussianMixture(k=k, seed=1, featuresCol="features", maxIter=50)
        model = gmm.fit(df_kmeans)
        output = model.transform(df_kmeans)
        score = evaluator.evaluate(output)
        print(f'k={k} -> sillhouette score={score}')
        wssse_values.append(score)

#run_lda()
run_bkmeans()
#run_gmm()
