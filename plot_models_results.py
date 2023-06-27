import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'AI Models': ['VGG16 model 1', 'VGG16 model 2', 'InceptionV3', 'ResNet50', 'Propriu 2D', 'Propriu 3D'],
    'Train Accuracy': [71.43, 61.39, 72.77, 54.73, 99.09, 73.62],
    'Validation Accuracy': [85.36, 66.95, 70.21, 51.30, 98.75, 74.46],
    'Test Accuracy': [84.96, 40.10, 76.89, 48.58, 87.91, 69.54]
}

df = pd.DataFrame(data)


def plot_grouped_bar_chart(df, title, ylabel):
    def add_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height - 2, f'{height:.2f}', ha='center', va='top')

    n_models = len(df['AI Models'])
    index = np.arange(n_models)
    bar_width = 0.25

    fig, ax = plt.subplots()
    train_bars = ax.bar(index, df['Train Accuracy'], bar_width, label='Acuratețe antrenare')
    val_bars = ax.bar(index + bar_width, df['Validation Accuracy'], bar_width, label='Acuratețe validare')
    test_bars = ax.bar(index + 2 * bar_width, df['Test Accuracy'], bar_width, label='Acuratețe testare')

    add_value_labels(train_bars, ax)
    add_value_labels(val_bars, ax)
    add_value_labels(test_bars, ax)

    ax.set_ylabel(ylabel)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(df['AI Models'])
    ax.set_ylim(0, 100)
    ax.legend()

    plt.show()


plot_grouped_bar_chart(df, 'AI Model Accuracies', 'Acuratețe (%)')
