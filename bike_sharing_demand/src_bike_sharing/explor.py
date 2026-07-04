
import pandas as pd

train = pd.read_csv("/Users/hamedmiri/Documents/Developer/ML/bike_sharing_demand/data/train.csv")
test = pd.read_csv("/Users/hamedmiri/Documents/Developer/ML/bike_sharing_demand/data/test.csv")

print(f"\nTrain Shape: {train.shape}")
print(f"\n test shape: {test.shape}")
print(f"\n train columns: {train.columns.tolist()}")
print(f"\n test columns: {test.columns.tolist()}")
print(f"\n First 5 rows of train dataset: {train.head()}")