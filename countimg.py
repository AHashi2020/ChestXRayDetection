import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

num_train = len(train_df)
num_val = len(val_df)
num_total = num_train + num_val

print("NIH Train:", num_train)
print("NIH Val:", num_val)
print("NIH Total Used:", num_total)