from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, size, broadcast
from pyspark.sql.types import ArrayType, FloatType, IntegerType
import ast

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, collect_list, sqrt
from pyspark.sql.functions import expr

from pyspark.ml.linalg import Vectors, VectorUDT


sc = SparkContext(appName="PROJECT")
sc.setLogLevel("ERROR")

spark = SparkSession \
    .builder \
    .appName("PROJECT") \
    .config("spark.dynamicAllocation.maxExecutors", "10") \
    .getOrCreate()

df_all = spark.read.json('/data/doina/WebInsight/*/*.gz')
relurls = spark.read.parquet('/user/s2350726/relurls.parquet')

# what is the schema of the web crawl dataframe?
# df_all.printSchema()

# df_all.select(df_all['url'], df_all['_corrupt_record']).groupby(df_all['_corrupt_record']).count()

count_elements = udf(lambda arr: len(arr), IntegerType())

# access 'url' as well as 'semanticVector' and 'fetchDate' under 'fetch'
df_all = df_all.select(
    df_all['url'],
    df_all['fetch.semanticVector'].alias('semanticVector'),
    df_all['fetch.fetchDate'].alias('fetchDate'),
    df_all['fetch.contentLength'].alias('contentLength'),
    df_all['fetch.internalLinks'].alias('internalLinks'),
    df_all['fetch.externalLinks'].alias('externalLinks')
    )
filtered_df = df_all.join(relurls, how="inner", on="url")

# how many rows are there in this dataframe?
# temp = df_all.count()
# print(f'Number of rows in the dataframe: {temp}')
# 10021263

# how many semantic vectors are 'not set'?
# temp = df_all.filter(df_all.semanticVector == 'not set').count()
# print(f'Number of semantic vectors that are not set: {temp}')
# 5581101

# what is the ratio of semantic vectors that are 'not set'?
# ratio = df_all.filter(df_all.semanticVector == 'not set').count() / df_all.count() * 100
# print(f'Ratio of semantic vectors that are not set: {ratio:.2f}%')
# 55.69 %

# how many semantic vectors are set?
# temp = df_all.filter(col('semanticVector').startswith("[")).count()
# print(f'Number of semantic vectors that are not set: {temp}')
# 44396938

# what is the ratio of semantic vectors that are set?
# ratio = df_all.filter(col('semanticVector').startswith("[")).count() / df_all.count() * 100
# print(f'Ratio of semantic vectors that are set: {ratio:.2f}%')
# 44.30 %


# Count the number of elements in each semanticVector and check if there is empty ones, just count them
# df_valid.filter(size(col('semanticVector')) == 0).count()
# Yay, there are no empty semantic vectors

# Additionally, count unique lengths of semanticVector
# df_valid.select(size(col('semanticVector'))).distinct().show()
# Yay, all the semantic vector lenghts are 192

# What are the number of fetchdates per url?
# df_valid.groupBy(['url']).count().collect()


# new changes :)

"""
>>> filtered_df.printSchema()
root
 |-- url: string (nullable = true)
 |-- semanticVector: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- fetchDate: string (nullable = true)

"""

## there is no need to use vectors, we can use array directly

# Convert the semanticVector array to a vector
# df_vectored = filtered_df.withColumn("semanticVector", col("semanticVector").cast(VectorAssembler().getOutputCol()))


#convert the semanticVector from string to list of floats
str_to_list = udf(lambda x: ast.literal_eval(x), ArrayType(FloatType()))
filtered_df = filtered_df.withColumn('semanticVector', str_to_list(filtered_df['semanticVector']))

filtered_df = filtered_df.withColumn("internalLinksCount", count_elements(col("internalLinks"))).withColumn("externalLinksCount", count_elements(col("externalLinks")))

# Create a Window specification partitioned by URL and ordered by fetchDate
window_spec = Window.partitionBy("url").orderBy("fetchDate")

# Use lag to shift the vectors by one position within each URL group
df_pairs = filtered_df.withColumn("previous_vector", lag("semanticVector").over(window_spec))

df_pairs = df_pairs.withColumn("previous_contentLength", lag("contentLength").over(window_spec))

df_pairs = df_pairs.withColumn("previous_internalLinksCount", lag("internalLinksCount").over(window_spec))

df_pairs = df_pairs.withColumn("previous_externalLinksCount", lag("externalLinksCount").over(window_spec))

# Drop the null values introduced by lag
df_pairs = df_pairs.na.drop(subset=["previous_vector"])

""" NO FUNCIONA
num_dimensions = 192

# Calculate Euclidean distance
euclidean_expression = sqrt(sum([(col('semanticVector')[i] - col('previous_vector')[i])**2 for i in range(num_dimensions)]))
df_pairs = df_pairs.withColumn("euclidean_distance", euclidean_expression)

"""

def euclidean_distance(v1, v2):
    return float(Vectors.dense(v1).squared_distance(Vectors.dense(v2)) ** 0.5)

def cosine_distance(v1, v2):
    v1_dense = Vectors.dense(v1)
    v2_dense = Vectors.dense(v2)
    return float(1 - v1_dense.dot(v2_dense) / (v1_dense.norm(2) * v2_dense.norm(2)))

def integer_difference(v1, v2):
    return abs(v2-v1)


udf_euclidean_distance = udf(euclidean_distance, FloatType())

udf_cosine_distance = udf(cosine_distance, FloatType())

udf_integer_difference = udf(integer_difference, IntegerType())


df_pairs = df_pairs.withColumn("euclidean_distance", udf_euclidean_distance("semanticVector", "previous_vector"))

df_pairs = df_pairs.withColumn("cosine_distance", udf_cosine_distance("semanticVector", "previous_vector"))

df_pairs = df_pairs.withColumn("content_length_difference", udf_integer_difference("previous_contentLength", "contentLength"))

df_pairs = df_pairs.withColumn("internal_link_count_difference", udf_integer_difference("previous_internalLinksCount", "internalLinksCount"))

df_pairs = df_pairs.withColumn("external_link_count_difference", udf_integer_difference("previous_externalLinksCount", "externalLinksCount"))

df_pairs = df_pairs.select("url","fetchDate","euclidean_distance","cosine_distance","content_length_difference","internal_link_count_difference","external_link_count_difference")

# Filter out rows where Euclidean distance is noht zero
# filtered_df_nonzero = df_pairs.filter((col("euclidean_distance") != 0.0) | (col("cosine_distance") != 0.0))
# Show the filtered DataFrame
# filtered_df_nonzero.show(10)

# >>> filtered_df_nonzero.count()
# 2343661                                                                         
# >>> df_pairs.count()
# 3455883

# max_distance_EUCLIDEAN_row = df_pairs.orderBy(col("euclidean_distance").desc()).first() # euclidean_distance=2.3577544689178467
# max_distance_COSINE_row = df_pairs.orderBy(col("cosine_distance").desc()).first()       # cosine_distance=1.1788772344589233
# max_distance_CONTENT_row = df_pairs.orderBy(col("content_length_difference").desc()).first()     # content_length_difference=1491113

# same vector for max_distance_EUCLIDEAN_row and max_distance_COSINE_row 

# max euclidean-distance = sqrt(4*192) is approximately 27.7 --> the largest Euclidean Distance in our space is approximately 2.4

content_length_difference = df_pairs.select("content_length_difference").rdd.flatMap(lambda x: x)
content_length_difference = content_length_difference.filter(lambda x: x is not None)
histogram_data = content_length_difference.histogram(20)
content_length_difference.unpersist()
for bin, count in zip(histogram_data[0], histogram_data[1]):
    print(f"Bin: {bin}, Count: {count}")
"""
Bin: 0.000000, Count: 3239587
Bin: 74555.65, Count: 1352
Bin: 149111.3, Count: 346
Bin: 223666.94999999998, Count: 188
Bin: 298222.6, Count: 94
Bin: 372778.25, Count: 45
Bin: 447333.89999999997, Count: 27
Bin: 521889.54999999993, Count: 19
Bin: 596445.2, Count: 26
Bin: 671000.85, Count: 12
Bin: 745556.5, Count: 9
Bin: 820112.1499999999, Count: 7
Bin: 894667.7999999999, Count: 5
Bin: 969223.45, Count: 2
Bin: 1043779.0999999999, Count: 0
Bin: 1118334.75, Count: 1
Bin: 1192890.4, Count: 2
Bin: 1267446.0499999998, Count: 0
Bin: 1342001.7, Count: 2
Bin: 1416557.3499999999, Count: 4
"""

euclidean_distance = df_pairs.select("euclidean_distance").rdd.flatMap(lambda x: x)
euclidean_distance = euclidean_distance.filter(lambda x: x is not None)
histogram_data = euclidean_distance.histogram(20)
euclidean_distance.unpersist()
for bin, count in zip(histogram_data[0], histogram_data[1]):
    print(f"Bin: {bin}, Count: {count}")
"""
Bin: 0.00000000000000000, Count: 3108303
Bin: 0.11788772344589234, Count: 77262
Bin: 0.23577544689178467, Count: 27022
Bin: 0.353663170337677, Count: 11973
Bin: 0.47155089378356935, Count: 6618
Bin: 0.5894386172294617, Count: 3467
Bin: 0.707326340675354, Count: 2289
Bin: 0.8252140641212463, Count: 1442
Bin: 0.9431017875671387, Count: 878
Bin: 1.060989511013031, Count: 526
Bin: 1.1788772344589233, Count: 328
Bin: 1.2967649579048157, Count: 303
Bin: 1.414652681350708, Count: 154
Bin: 1.5325404047966005, Count: 110
Bin: 1.6504281282424926, Count: 89
Bin: 1.768315851688385, Count: 166
Bin: 1.8862035751342774, Count: 362
Bin: 2.0040912985801698, Count: 327
Bin: 2.121979022026062, Count: 91
Bin: 2.2398667454719545, Count: 18
"""

cosine_distance = df_pairs.select("cosine_distance").rdd.flatMap(lambda x: x)
cosine_distance = cosine_distance.filter(lambda x: x is not None)
histogram_data = cosine_distance.histogram(20)
cosine_distance.unpersist()
for bin, count in zip(histogram_data[0], histogram_data[1]):
    print(f"Bin: {bin}, Count: {count}")
"""
Bin: -2.220446049250313e-16, Count: 3108303
Bin: 0.05894386172294595, Count: 77262
Bin: 0.11788772344589213, Count: 27022
Bin: 0.1768315851688383, Count: 11973
Bin: 0.23577544689178448, Count: 6618
Bin: 0.29471930861473067, Count: 3467
Bin: 0.3536631703376768, Count: 2289
Bin: 0.412607032060623, Count: 1442
Bin: 0.4715508937835692, Count: 878
Bin: 0.5304947555065154, Count: 526
Bin: 0.5894386172294616, Count: 328
Bin: 0.6483824789524077, Count: 303
Bin: 0.7073263406753538, Count: 154
Bin: 0.7662702023983, Count: 110
Bin: 0.8252140641212462, Count: 89
Bin: 0.8841579258441924, Count: 166
Bin: 0.9431017875671386, Count: 362
Bin: 1.0020456492900847, Count: 327
Bin: 1.060989511013031, Count: 91
Bin: 1.119933372735977, Count: 18
"""
cosine_distance = df_pairs.select("cosine_distance").rdd.flatMap(lambda x: x)
cosine_distance = cosine_distance.filter(lambda x: x is not None)
# Filtered to only consider nonnegative values
cosine_distance = cosine_distance.filter(lambda x: x >= 0)
histogram_data = cosine_distance.histogram(20)
cosine_distance.unpersist()
for bin, count in zip(histogram_data[0], histogram_data[1]):
    print(f"Bin: {bin}, Count: {count}")
"""
Bin: 0.00000000000000000, Count: 2596711
Bin: 0.05894386172294617, Count: 77262
Bin: 0.11788772344589234, Count: 27022
Bin: 0.1768315851688385, Count: 11973
Bin: 0.23577544689178467, Count: 6618
Bin: 0.29471930861473083, Count: 3467
Bin: 0.353663170337677, Count: 2289
Bin: 0.41260703206062316, Count: 1442
Bin: 0.47155089378356935, Count: 878
Bin: 0.5304947555065155, Count: 526
Bin: 0.5894386172294617, Count: 328
Bin: 0.6483824789524079, Count: 303
Bin: 0.707326340675354, Count: 154
Bin: 0.7662702023983002, Count: 110
Bin: 0.8252140641212463, Count: 89
Bin: 0.8841579258441925, Count: 166
Bin: 0.9431017875671387, Count: 362
Bin: 1.0020456492900849, Count: 327
Bin: 1.060989511013031, Count: 91
Bin: 1.1199333727359773, Count: 18
"""

# It seems that there are many zero values. There is a long tail on the positive distance values. 

print(df_pairs.count())
# In total, there are 3,241,728 records in df_pairs.

# How many records have zero values for the content distance?
content_zero_count = df_pairs.filter(col("content_length_difference") == 0).count()
print(f"There are {content_zero_count} records with no change in contentLength between two consecutive weeks.")
# There are 1686308 records with no change in contentLength between two consecutive weeks.


# How many records have zero values for the euclidean distance?
euclid_zero_count = df_pairs.filter(col("euclidean_distance") == 0).count()
print(f"There are {euclid_zero_count} records where the semanticVectors of two consecutive weeks have a zero Euclidean distance (the vectors are identical).")
# There are 1957404 records where the semanticVectors of two consecutive weeks have a zero Euclidean distance (the vectors are identical).


# How many records have zero values for the cosine distance?
cosine_zero_count = df_pairs.filter(col("cosine_distance") == 0).count()
print(f"There are {cosine_zero_count} records where the semanticVectors of two consecutive weeks have a zero cosine distance (their angle is 0 degrees).")
# There are 1015401 records where the semanticVectors of two consecutive weeks have a zero cosine distance (their angle is 0 degrees).

# What is the correlation between the differences of some variables on the one hand and the euclidean - and cosine distance on the other hand?
df_pairs.write.mode("overwrite").parquet("df_pairs.parquet")

# What is the embedding logic? Do we average over or sum the word count in a topic or feature?

# Discuss the expected relationship between Euclidean and cosine distance.
 
# Remove all records where content distance is zero.
df_nonzero = df_pairs.filter(col("content_length_difference") > 0)

# Plot the Euclidean distance and cosine distance as a function of the content distance on a scatter plot. Performed this on my laptop rather than cluster.


####

# from pyspark.ml.clustering import KMeans
# from pyspark.ml.linalg import Vectors, VectorUDT
# from pyspark.ml.feature import VectorAssembler

# #TODO: check if we can pass a list to the VectorAssembler instead of multiple columns containing single values
# list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
# df_valid = df_valid.withColumn('semanticVector', list_to_vector_udf(df_valid['semanticVector']))

# #Check if there are null values in the semanticVector column
# df_valid.filter(df_valid.semanticVector.isNull()).count()

# # Convert the semanticVector column to a Vector type
# vecAssembler = VectorAssembler(inputCols=["semanticVector"], outputCol="features")
# df_kmeans = vecAssembler.transform(df_valid)

# # Initialize the k-means algorithm
# kmeans = KMeans(k=10, seed=1, maxIter=10, featuresCol="features")  # change k to the desired number of clusters
# model = kmeans.fit(df_kmeans)
# df_kmeans = model.transform(df_kmeans)

# df_kmeans.show()

# df.write.mode("overwrite").csv("WEB")