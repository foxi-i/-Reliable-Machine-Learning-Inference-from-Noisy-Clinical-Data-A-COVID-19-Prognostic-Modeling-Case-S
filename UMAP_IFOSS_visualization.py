
"""
UMAP Visualization of IFOSS

This script visualizes the effect of the IFOSS preprocessing
framework using supervised UMAP.

Subplots
--------
1. Original training dataset
2. Training dataset after IFOSS

Required variables
------------------
X_train : pandas DataFrame
    Original training features.

y_train : pandas Series or NumPy array
    Original training labels.

X_resampled : pandas DataFrame
    Training features after IFOSS preprocessing.

y_resampled : pandas Series or NumPy array
    Training labels after IFOSS preprocessing.

Requirements
------------
Tested with:

    scikit-learn==1.5.2
    umap-learn==0.5.6

If compatibility errors involving `check_array`
occur, install compatible versions of
scikit-learn and umap-learn.

Output
------
A two-panel UMAP visualization comparing the
original training dataset with the IFOSS
preprocessed dataset.
"""

import numpy as np
import umap
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(10, 5)

)

plot_colors = [

    "green",

    "red"

]

fig.suptitle(

    "Training Dataset Before and After IFOSS",

    fontsize=14,

    fontweight="bold"

)

# ============================================================
# ORIGINAL TRAINING DATASET
# ============================================================

reducer_original = umap.UMAP(

    n_neighbors=10,

    n_epochs=100,

    learning_rate=0.0009,

    random_state=42,

    transform_seed=42,

    verbose=False

)

reducer_original.fit(

    X_train,

    y_train

)

embedding_original = reducer_original.transform(

    X_train

)

for i, label in enumerate(y_train):

    axes[0].scatter(

        embedding_original[i, 0],

        embedding_original[i, 1],

        color=plot_colors[label],

        s=10

    )

axes[0].set_title(

    "Original Training Dataset",

    fontsize=12,

    fontweight="bold"

)

# ============================================================
# IFOSS PREPROCESSED DATASET
# ============================================================

reducer_ifoss = umap.UMAP(

    n_neighbors=10,

    n_epochs=100,

    learning_rate=0.0009,

    random_state=42,

    transform_seed=42,

    verbose=False

)

reducer_ifoss.fit(

    X_resampled,

    y_resampled

)

embedding_ifoss = reducer_ifoss.transform(

    X_resampled

)

for i, label in enumerate(y_resampled):

    axes[1].scatter(

        embedding_ifoss[i, 0],

        embedding_ifoss[i, 1],

        color=plot_colors[label],

        s=10

    )

axes[1].set_title(

    "Training Dataset After IFOSS",

    fontsize=12,

    fontweight="bold"

)


# ============================================================
# LEGEND
# ============================================================

legend_handles = [

    plt.Line2D(

        [0],

        [0],

        marker="o",

        color="w",

        markerfacecolor="green",

        markersize=8,

        label="Negative Class"

    ),

    plt.Line2D(

        [0],

        [0],

        marker="o",

        color="w",

        markerfacecolor="red",

        markersize=8,

        label="Positive Class"

    )

]

fig.legend(

    handles=legend_handles,

    loc="lower center",

    ncol=2,

    frameon=False,

    bbox_to_anchor=(0.5, -0.02)

)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(

    rect=[0, 0.05, 1, 0.95]

)

plt.show()


