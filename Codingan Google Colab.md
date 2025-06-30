#   GOOGLE-COLLAB
!pip install yellowbrick
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
import warnings; warnings.filterwarnings('ignore')
df = pd.read_csv("marketing_campaign.csv", sep="\t")
df.dropna(inplace=True)
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
df["Customer_For"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
df["Age"] = 2021 - df["Year_Birth"]
df["Spent"] = df[[
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]].sum(axis=1)
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Family_Size"] = df["Children"] + df["Marital_Status"].replace({
    "Single": 1, "Married": 2, "Together": 2, "Divorced": 1, "Widow": 1, "Alone": 1, "Absurd": 1, "YOLO": 1
}).fillna(1).astype(int)
df.drop(["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"], axis=1, inplace=True)
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
})

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
visualizer = KElbowVisualizer(KMeans(), k=(2,10))
visualizer.fit(pca_df)
visualizer.show()
ac = AgglomerativeClustering(n_clusters=4)
clusters = ac.fit_predict(pca_df)
df["Cluster"] = clusters
sns.scatterplot(x=df["Income"], y=df["Spent"], hue=df["Cluster"], palette="Set2")
plt.title("Customer Cluster by Income and Spending")
plt.show()
