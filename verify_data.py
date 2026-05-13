import pandas as pd
import numpy as np

file_path = r'c:\Users\evlat\OneDrive\Desktop\makine öğrenmesi\Covid Data.csv'

print("Loading data...")
df = pd.read_csv(file_path)

print(f"Original shape: {df.shape}")

# Feature Engineering: Create Target Variable 'DEATH'
df['DEATH'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)
df.drop('DATE_DIED', axis=1, inplace=True)

# Data Cleaning
df['INTUBED'] = df['INTUBED'].replace([97, 98, 99], 2)
df['ICU'] = df['ICU'].replace([97, 98, 99], 2)
df['PREGNANT'] = df['PREGNANT'].replace([97, 98, 99], 2)

cols_to_check = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 
                 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']

for col in cols_to_check:
    df = df[df[col] < 98]

print(f"Cleaned shape: {df.shape}")
print("\nTarget distribution (DEATH):")
print(df['DEATH'].value_counts())
print("\nFirst 5 rows:")
print(df.head())
