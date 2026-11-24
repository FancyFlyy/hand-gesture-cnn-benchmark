import matplotlib.pyplot as plt
import numpy as np


# ======================
# EVALUATION RESULTS
# ======================
# Urutan model harus sama

models = [
    "RESNET18",
    "MNV2",
    "GHOSTNET",
    "MNV3"
]


# ======================
# HGRID RESULTS
# ======================
hgrid_acc = [
    0.9828,
    0.9820,
    0.9691,
    0.9445
]


# ======================
# ASL RESULTS
# ======================
asl_acc = [
    0.9996,
    1.0000,
    0.9967,
    0.9967
]


# ======================
# BAR POSITION
# ======================
x = np.arange(
    len(models)
)

bar_width = 0.32


# ======================
# FIGURE
# ======================
plt.figure(
    figsize=(5, 3.5)
)


# ======================
# BAR CHART
# ======================
bars1 = plt.bar(
    x - bar_width/2,
    hgrid_acc,
    width=bar_width,
    label="HaGRID"
)

bars2 = plt.bar(
    x + bar_width/2,
    asl_acc,
    width=bar_width,
    label="ASL"
)


# ======================
# CLEAN STYLE
# ======================
ax = plt.gca()

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ======================
# AXIS RANGE
# ======================
plt.ylim(
    0.90,
    1.01
)


# ======================
# LABELS
# ======================
plt.ylabel(
    "Accuracy",
    fontsize=8
)

plt.title(
    "CNN Model Performance Across Datasets",
    fontsize=9
)

plt.xticks(
    x,
    models,
    fontsize=7
)

plt.yticks(
    fontsize=7
)


# ======================
# LEGEND
# ======================
plt.legend(
    fontsize=7,
    loc="lower right"
)


# ======================
# VALUE LABELS
# ======================
for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        plt.text(
            bar.get_x()
            + bar.get_width()/2,

            height + 0.0015,

            f"{height:.4f}",

            ha="center",

            fontsize=6
        )


# ======================
# SAVE
# ======================
plt.tight_layout()

plt.savefig(
    "cross_dataset_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# ======================
# SHOW
# ======================
plt.show()


print(
    "\nSaved:"
)

print(
    "cross_dataset_model_comparison.png"
)