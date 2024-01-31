from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hash, monotonically_increasing_id, udf, explode
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import split, regexp_replace, isnan, when, count, col
from pyspark.ml.linalg import Vectors, VectorUDT
from torch import nn
from pyspark.ml.torch.distributor import TorchDistributor
import numpy as np
import os
from torch.utils.data import Dataset

#spark = SparkSession.builder.getOrCreate()
spark = SparkSession.builder.config("spark.dynamicAllocation.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print('Imports done')

# df = spark.read.json('/data/doina/WebInsight/*/*.gz') # all data
df = spark.read.json('/data/doina/WebInsight/2020-09-07/*.gz')  # one week of data
df = df.limit(1000)

print('Loaded data')

# Get semanticVector and url
df = df.select(df['url'], df['fetch.semanticVector'].alias('semanticVector'))
df = df.filter(col('semanticVector').startswith("["))
df = df.withColumn('semanticVector', split(regexp_replace('semanticVector', r'(\[|\])', ''), ', '))
df = df.withColumn('semanticVector', df['semanticVector'].cast(ArrayType(FloatType())))

# Convert list of floats to dense vector
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn('semanticVector', list_to_vector_udf(df['semanticVector']))

# Semantic vector has 192 elements in each vector, and we aim to reduce it to X elements using an autoencoder

df_train = df.withColumn('label', df['semanticVector']) # Convert semanticVector to label for training
df_train = df_train.withColumnRenamed('semanticVector', 'features')


#TODO: implement https://learn.microsoft.com/en-us/azure/databricks/machine-learning/train-model/distributed-training/spark-pytorch-distributor

class SparkDataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_col, label_col):
        """
        Args:
            dataframe (Spark DataFrame): Spark DataFrame containing the features and labels.
            feature_col (str): Name of the column containing the features.
            label_col (str): Name of the column containing the labels.
        """
        # Collect dataframe to local machine, might not be feasible for large datasets
        self.features = dataframe.select(feature_col).rdd.flatMap(lambda x: x).collect()
        self.labels = dataframe.select(label_col).rdd.flatMap(lambda x: x).collect()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

from pyspark.ml.torch.distributor import TorchDistributor
import torch
import torch.distributed as dist
import torch.nn.parallel.DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 192),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, loader, learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(loader)}')

    print('Finished Training')
    return model

def distributed_train(learning_rate, use_gpu):
    import torch
    import torch.distributed as dist
    import torch.nn.parallel.DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler, DataLoader

    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend)
    device = int(os.environ["LOCAL_RANK"]) if use_gpu  else "cpu"
    model = DDP(Autoencoder())
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler)

    output = train(model, loader, learning_rate)
    dist.cleanup()
    return output

gpu = False
dataset = SparkDataFrameDataset(df_train, 'features', 'label')
distributor = TorchDistributor(num_processes=10, local_mode=False, use_gpu=gpu)
distributor.run(distributed_train, 1e-3, gpu)




# model = Autoencoder()
#
# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Convert Spark DataFrame to PyTorch DataLoader

# features = df_train.select('features').rdd.flatMap(lambda x: x).collect()
# labels = df_train.select('label').rdd.flatMap(lambda x: x).collect()
# dataset = list(zip(features, labels))
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#
# # Use TorchDistributor for distributed training
# ...
#
# # After training, use the model to transform your data and calculate the reconstruction error
# ...
#
# # Calculate the reconstruction error
# mse = np.mean((result - labels)**2)
# print("Mean Squared Error = " + str(mse))