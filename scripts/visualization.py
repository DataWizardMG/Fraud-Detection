import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df, cat_cols, num_cols):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i, subplot in zip(cat_cols, ax.flatten()):
        sns.countplot(data=df, x=i, hue='Is Fraudulent', ax=subplot, palette='BuPu')
        subplot.tick_params(axis='x', labelrotation=90)
        subplot.get_legend().remove()
    plt.tight_layout()
    handles, _ = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, ['Not Fraudulent', 'Fraudulent'], ncol=2, bbox_to_anchor=(0.68, 1.05))
    plt.show()

    fig, ax = plt.subplots(4, 3, figsize=(12, 12))
    for i, subplot in zip(num_cols, ax.flatten()):
        sns.histplot(data=df, x=i, hue='Is Fraudulent', bins='auto', ax=subplot, palette='BuPu')
        subplot.get_legend().remove()
    fig.delaxes(ax[3, 2])
    plt.tight_layout()
    fig.legend(handles, ['Not Fraudulent', 'Fraudulent'], ncol=2, bbox_to_anchor=(0.68, 1.05))
    plt.show()

    sns.heatmap(df[num_cols].corr(), annot=True, cmap="crest", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()