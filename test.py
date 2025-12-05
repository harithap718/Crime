import pandas as pd

# Load first 50,000 rows only
df = pd.read_csv(r"D:\crime_project2\cleaned_crimes.csv", nrows=50000)

# Save this smaller file
df.to_csv("cleaned_sample_50k.csv", index=False)

print("Saved cleaned_sample_50k.csv successfully!")

