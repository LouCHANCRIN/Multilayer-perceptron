import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def main():
    if (len(sys.argv) <= 1):
        sys.exit("No name file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")

    labels = []
    for i in range(0, 32):
        labels.append(f"feature{i}")
    ressource = sys.argv[1]
    try:
        data = pd.read_csv(ressource, names=labels)
    except:
        sys.exit(f"Failed to read file {sys.argv[1]}")
    sns.pairplot(data, hue="feature1")
    plt.show()

if __name__ == "__main__":
    main()