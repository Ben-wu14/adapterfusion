import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from tqdm import tqdm

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data[i, j]!=0.0:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def heat_sta(path, tasks):
    path += '/' if not path.endswith('/') else ""
    imp = []
    for k in tqdm(tasks):
        task_imp = [0.0]*24
        with open(path+k+'/importance.pkl', 'rb') as f:
            data = pickle.load(f)
            for key in data:
                avg_imp = data[key]['importance'][1]*100
                module, layer = key
                if module == 'output':
                    task_imp[layer*2+1] = avg_imp
                else:
                    task_imp[layer*2] = avg_imp
        imp.append(task_imp)

    layer_name = []
    for i in range(12):
        layer_name.append('att-'+str(i+1))
        layer_name.append('out-'+str(i+1))



    fig, ax = plt.subplots()

    im, cbar = heatmap(np.array(imp), list(tasks.keys()), layer_name, ax, cmap="Oranges", cbarlabel="IMP [percent/layer]")
    texts = annotate_heatmap(im, valfmt="{x:.1f}%")

    fig.tight_layout()
    plt.show()


def bars_sta(path, tasks):
    path += '/' if not path.endswith('/') else ""
    ori_imp = []
    lay_imp = []
    for k in tqdm(tasks):

        with open(path+"/Origin/"+k+'/importance.pkl', 'rb') as f:
            data = pickle.load(f)
            task_imp = np.array([data[key]['importance'][1]*100 for key in data])
        ori_imp.append(task_imp.mean())

        
        with open(path+"/Layer/"+k+'/importance.pkl', 'rb') as f:
            data = pickle.load(f)
            task_imp = np.array([data[key]['importance'][1]*100 for key in data])
        lay_imp.append(task_imp.mean())

    x = np.arange(len(list(tasks.keys())))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, ori_imp, width, label='Origin model')
    rects2 = ax.bar(x + width/2, lay_imp, width, label='Layer model')

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    ax.set_xticks(x)
    ax.set_xticklabels(list(tasks.keys()))

    ax.set_title('Adapter importance')
    ax.legend()
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Tasks')
    fig.tight_layout()

    plt.show()

tasks = {
        'cola':8.5,
        'sst2':67,
        'mrpc':3.7,
        'stsb':7,
        'qqp':364,
        'mnli':393,
        'mnli-mm':393,
        'qnli':105,
        'rte':2.5,
    }
tasks = {k:v for k,v in sorted(tasks.items(), key=lambda item: item[1])}
path = 'results/ST-A/Origin'
bars_sta(path, tasks)