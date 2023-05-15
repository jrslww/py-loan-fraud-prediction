import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(data):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.show()
