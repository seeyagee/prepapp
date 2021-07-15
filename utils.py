import matplotlib.pyplot as plt
import numpy as np


def show_hbar(names, values, title, kind='relative'):

    if kind == 'relative':
        values = (values/ sum(values)).round(2)

    fig, ax = plt.subplots()

    y_pos = np.arange(len(values))

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title(title)

    return fig