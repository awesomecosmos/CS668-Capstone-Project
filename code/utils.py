import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, colname, data_dict):
    # setting variables
    desc = data_dict[colname]['var_desc_short']
    vals = data_dict[colname]['data_values']
    # plotting figure
    plt.figure(figsize=(10,5))
    sns.histplot(df[colname])
    plt.xlabel(f'Value of {colname} ({vals})')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of values in {colname} ({desc})')
    plt.savefig(f'../img/eda/{colname}_hist.jpeg',dpi=900)
    plt.show()