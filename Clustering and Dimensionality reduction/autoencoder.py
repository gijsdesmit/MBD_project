from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hash, monotonically_increasing_id, udf, explode
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import split, regexp_replace, when, count, col
from pyspark.ml.linalg import Vectors, VectorUDT
import torch
from torch import nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

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

from sklearn.preprocessing import MinMaxScaler
import pickle
# Convert the list of floats to a nested list of floats
df_train_pd['features'] = df_train_pd['features'].apply(lambda x: [[i] for i in x])
# Initialize a scaler with feature range -1 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler to the features and transform
df_train_pd['features'] = df_train_pd['features'].apply(lambda x: scaler.fit_transform(x))
# Flatten the nested list of floats back to a list of floats
df_train_pd['features'] = df_train_pd['features'].apply(lambda x: [item for sublist in x for item in sublist])
# Save the scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

s1 = 192
s2 = 128
s3 = 64

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 192 ==> 128 ==> 64
        self.encoder = nn.Sequential(
            nn.Linear(s1, s2),
            nn.ReLU(),
            nn.Linear(s2, s3),
            nn.ReLU()
        )
        # 64 ==> 128 ==> 192
        self.decoder = nn.Sequential(
            nn.Linear(s3, s2),
            nn.ReLU(),
            nn.Linear(s2, s1),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        features = torch.tensor(self.data.iloc[index]['features'], dtype=torch.float)
        label = torch.tensor(self.data.iloc[index]['features'], dtype=torch.float)
        return features, label

print('Converting to local dataset')
dataset = CustomDataset(df_train_pd)
dataloader = DataLoader(dataset, batch_size=80, shuffle=True)

print('Creating model')
model = AutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
epochs = 20
losses = []

print('Starting training')
for epoch in range(epochs):
    for (semantic_vector, _) in dataloader:
        # Output of Autoencoder
        reconstructed = model(semantic_vector)
        # Calculating the loss function
        loss = loss_function(reconstructed, semantic_vector)
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Storing the losses in a list for plotting
        losses.append(loss.item())
        #show 1 reconstructed vector vs original
    #print('Original:', semantic_vector[0])
    #print('Reconstructed:', reconstructed[0])
    print(f'Epoch:{epoch + 1}, Loss:{loss.item()}')

