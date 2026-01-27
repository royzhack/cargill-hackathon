import pandas as pd

# Convert Cargill_Committed_Cargoes_Structured.xlsx to CSV
print("Converting Cargill_Committed_Cargoes_Structured.xlsx to CSV...")
df1 = pd.read_excel("data/Cargill_Committed_Cargoes_Structured.xlsx")
df1.to_csv("data/Cargill_Committed_Cargoes_Structured.csv", index=False)
print("✓ Created Cargill_Committed_Cargoes_Structured.csv")

# Convert Market_Cargoes_Structured.xlsx to CSV
print("Converting Market_Cargoes_Structured.xlsx to CSV...")
df2 = pd.read_excel("data/Market_Cargoes_Structured.xlsx")
df2.to_csv("data/Market_Cargoes_Structured.csv", index=False)
print("✓ Created Market_Cargoes_Structured.csv")

print("\nConversion complete!")
