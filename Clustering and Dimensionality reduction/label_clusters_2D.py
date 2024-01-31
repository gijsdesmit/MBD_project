from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import col, explode, lower, udf
from pyspark.sql.types import ArrayType, StringType
import re

# Initialize Spark Session
spark = SparkSession.builder.appName("ClusterKeywordsExtraction").getOrCreate()

# Sample DataFrame
df = spark.read.csv('file:/home/s2559811/dbs_results.csv', header=True) # contains url, Dimension 1, Dimension 2

# Define a UDF to preprocess URLs: remove protocols, domain names, and split compound words
def preprocess_url(url):
    #remove .html and .htm at the end
    url = re.sub(r"\.html?$", "", url)
    #remove everything after the last dot
    url = re.sub(r"\.[^\.]*$", "", url)
    tokens = re.sub(r"https?://|www\.", "", url).split('/') # Remove protocol and split by non-alphanumeric characters
    split_tokens = [re.split(r'\W+', token) for token in tokens if token] # Further split on non-alphanumeric characters to separate compound words
    flat_tokens = [item for sublist in split_tokens for item in sublist if item]
    return flat_tokens

preprocess_url_udf = udf(preprocess_url, ArrayType(StringType()))

# Tokenize and preprocess URLs
df = df.withColumn("tokens", preprocess_url_udf(col("url")))
# Remove stopwords
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
df = remover.transform(df)

# Explode tokens to count occurrences
df = df.withColumn("token", explode(col("filtered"))).select("prediction", "token")
df = df.withColumn("token", lower(col("token")))

# Count token frequency within each cluster
df_keyword_frequency = df.groupby("prediction", "token").count()

# For each cluster, extract top N keywords
# This part requires a Window function and ROW_NUMBER to rank keywords by frequency within each cluster
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

windowSpec = Window.partitionBy(df_keyword_frequency['prediction']).orderBy(df_keyword_frequency['count'].desc())

# Add row number within each partition
df_keywords_ranked = df_keyword_frequency.withColumn("rank", row_number().over(windowSpec))

# Filter for top N (e.g., top 5) keywords for each cluster
top_n = 1
df_top_keywords = df_keywords_ranked.filter(col("rank") <= top_n).select("prediction", "token", "count")
df_top_keywords = df_top_keywords.sort(df_top_keywords.prediction.asc())
df_top_keywords.show()


