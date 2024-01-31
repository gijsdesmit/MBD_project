import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_parquet("./df_pairs.parquet")

# print(df.describe())

# Count the number of URLs with zero change in contentLength
# temp = df.groupby("url")["content_length_difference"].mean()
# print(f"There are {len(temp[temp == 0])} URLs with zero content length change over the 10 weeks")
# There are 84863 URLs with zero content length change over the 10 weeks


# Count the number of URLs with the same semantic vector for all 10 weeks
# temp = df.groupby("url")["euclidean_distance"].mean()
# print(f"There are {len(temp[temp == 0])} URLs with the same semantic vector for all 10 weeks")
# There are 122963 URLs with the same semantic vector for all 10 weeks


# plt.figure(figsize=(8, 6))
# plt.plot(df['content_length_difference'], df['euclidean_distance'], marker='.', linestyle='None', label='Euclidean Distance')
# plt.plot(df['content_length_difference'], df['cosine_distance'], marker='.', linestyle='None', label='Cosine Distance')
# plt.xlabel('Content Length Difference')
# plt.ylabel('Distance')
# plt.title('Cosine and Euclidean Distance vs Content Length Difference')
# plt.legend()
# plt.grid()
# plt.show()


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
# ax1.plot(df['internal_link_count_difference'], df['euclidean_distance'], marker='.', linestyle='None', label='Euclidean Distance')
# ax1.plot(df['internal_link_count_difference'], df['cosine_distance'], marker='.', linestyle='None', label='Cosine Distance')
# ax1.set_xlabel('Internal Link Count Difference')
# ax1.set_ylabel('Distance Travelled')
# ax1.grid()
# ax2.plot(df['external_link_count_difference'], df['euclidean_distance'], marker='.', linestyle='None', label='Euclidean Distance')
# ax2.plot(df['external_link_count_difference'], df['cosine_distance'], marker='.', linestyle='None', label='Cosine Distance')
# ax2.set_xlabel('External Link Count Difference')
# ax2.grid()
# ax3.plot(df['content_length_difference'], df['euclidean_distance'], marker='.', linestyle='None', label='Euclidean Distance')
# ax3.plot(df['content_length_difference'], df['cosine_distance'], marker='.', linestyle='None', label='Cosine Distance')
# ax3.set_xlabel('Content Length Difference')
# ax3.grid()
# # plt.title('Cosine and Euclidean Distance Travelled vs Content Length Difference')
# plt.legend()
# plt.show()

# n = df.count()
# print(f"The dataset contains {len(df)} records from {df['url'].nunique()} unique webpages. The ratio of zero to nonzero values for the numerical variables is as follows:")
# print(((df == 0).mean() * 100).round(1))

"""
\begin{table}[]
\begin{tabular}{|l|l|}
\hline
\textbf{column name}           & \textbf{\begin{tabular}[c]{@{}l@{}}percentage of\\ zero records\end{tabular}} \\ \hline
webpage url                    & 00.0                                                                          \\ \hline
fetch date                     & 00.0                                                                          \\ \hline
Internal link count difference & 83.2                                                                          \\ \hline
external link count difference & 94.5                                                                          \\ \hline
content length difference      & 52.0                                                                          \\ \hline
Euclidean distance             & 60.4                                                                          \\ \hline
cosine distance                & 31.3                                                                          \\ \hline
\end{tabular}
\end{table}
"""

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
plt1 = df['internal_link_count_difference'].value_counts().sort_index()
ax1.plot(plt1, color='C2', marker='.', linestyle='')
ax1.set_xlabel('Internal Link Count Difference')
ax1.set_ylabel('Number of records')
plt2 = df['external_link_count_difference'].value_counts().sort_index()
ax2.plot(plt2, color='C3', marker='.', linestyle='None')
ax2.set_xlabel('External Link Count Difference')
plt3 = df['content_length_difference'].value_counts().sort_index()
ax3.plot(plt3, color='C4', marker='.', linestyle='None')
ax3.set_xlabel('Content Length Difference')
plt4 = df['euclidean_distance'].value_counts().sort_index()
ax4.plot(plt4, color='C0', marker='.', linestyle='None')
ax4.set_xlabel('Euclidean Distance')
plt5 = df['cosine_distance'].value_counts().sort_index()
ax5.plot(plt5, color='C1', marker='.', linestyle='None')
ax5.set_xlabel('Cosine Distance')
ax3.xaxis.label.set_size(8)
plt.show()

# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
# ax1.set_xlim((0,100))
# ax1.hist(df['internal_link_count_difference'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# ax1.set_xlabel('Internal Link Count Difference')
# ax2.set_xlim((0,100))
# ax2.hist(df['external_link_count_difference'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# ax2.set_xlabel('External Link Count Difference')
# ax3.set_xlim((0,100000))
# ax3.hist(df['content_length_difference'], bins=[i*1000 for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
# ax3.set_xlabel('Content Length Difference')
# ax4.hist(df['euclidean_distance'])
# ax4.set_xlabel('Euclidean Distance')
# ax5.hist(df['cosine_distance'])
# ax5.set_xlabel('Cosine Distance')
# ax3.xaxis.label.set_size(8)
# plt.ylim((0,0.40*10**6))
# plt.show()


# import seaborn as sns
# plt.figure(figsize=(8, 6))
# sns.kdeplot(x=df['content_length_difference'], y=df['cosine_distance'], cmap="Reds", fill=True, label='Cosine Distance')
# sns.kdeplot(x=df['content_length_difference'], y=df['euclidean_distance'], cmap="Blues", fill=True, label='Euclidean Distance')
# plt.xlabel('Content Distance')
# plt.ylabel('Density')
# plt.title('Density Plot of Cosine and Euclidean Distance vs Content Distance')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.hexbin(x=df["content_length_difference"], y=df["cosine_distance"])
# plt.show()

# print(df['content_length_difference'].isna().sum())
# print(df['cosine_distance'].isna().sum())
# print(df['euclidean_distance'].isna().sum())
# print(df['internal_link_count_difference'].isna().sum())
# print(df['external_link_count_difference'].isna().sum())

# print(df.corr("spearman"))

"""
    Pearson's correlation coefficient (PCC) measures linear correlation between two sets of data.
    It is the ratio between the covariance of two variables and the product of their standard deviations;
    thus, it is essentially a normalized measurement of the covariance, such that the result always has a value between −1 and 1.
"""
# Corelation matrix with statistical significance denoted in asteriks by tozCSS from https://stackoverflow.com/a/49040342
# from scipy.stats import pearsonr
# import numpy as np

# def plot_corr(df):
#     rho = df.corr()
#     pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
#     p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
#     return rho.round(2).astype(str) + p

# print(plot_corr(df))