import pandas as pd
import matplotlib.pyplot as plt

# Load the data we generated in experiment_spartan_v2.py
df = pd.read_csv("bmoi_spartan_results.csv")

plt.figure(figsize=(10, 6))

# Plot Naive Curve
naive_data = df[df["Type"] == "Naive"]
plt.plot(naive_data["Damage"], naive_data["Fitness"], 
         marker='o', linestyle='--', color='red', label='Naive (Standard AI)')

# Plot Spartan Curve
spartan_data = df[df["Type"] == "Spartan"]
plt.plot(spartan_data["Damage"], spartan_data["Fitness"], 
         marker='o', linewidth=3, color='green', label='Spartan (Antifragile AI)')

plt.title("The Antifragility Gap: Performance Under Fire", fontsize=14)
plt.xlabel("Structural Damage (Probability of Cell Death)", fontsize=12)
plt.ylabel("Cognitive Function (Cosine Similarity)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Save for LaTeX
plt.savefig("bmoi_spartan_curve.png", dpi=300)
print("✅ Chart generated: bmoi_spartan_curve.png")