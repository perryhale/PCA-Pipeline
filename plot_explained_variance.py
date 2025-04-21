import numpy as np
import matplotlib.pyplot as plt
import glob


# define keys
file_paths = glob.glob("*.npz")

# define colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(file_paths)))

# determine cumulative explained variance ratios
cumulative_ratios = {}
for path in file_paths:
	key = path.replace('pca_', '').replace('.npz', '')
	data = np.load(path)
	ratio = data['explained_variance_ratio']
	cumulative_ratios[key] = np.cumsum(ratio)

# init plot
fig, axis = plt.subplots(nrows=2, ncols=1, figsize=(9,10))
ax0, ax1 = axis

# plot cev ratios
for i, (key, cum_ratio) in enumerate(cumulative_ratios.items()):
	color = colors[i % len(colors)]
	for ax in axis:
		ax.plot(np.arange(1, len(cum_ratio) + 1), cum_ratio, label=key, color=color)

# set common properties
for ax in axis:
	ax.grid()
	y = np.linspace(0, 1, 11)
	ax.set_yticks(y, [f'{yi:.2f}' for yi in y])

# set per axis properties
ax1.set_xlim(0,128)
ax0.set_ylabel("Cumulative variance ratio")
ax1.set_ylabel("Cumulative variance ratio")
ax1.set_xlabel("Principal axis")

# draw annotations
#ax1.axhline(0.95, c='Red', linestyle='dashed', label='95% Threshold')
handles, labels = ax1.get_legend_handles_labels()
ax0.legend(handles, labels,
	bbox_to_anchor=(1.05, 1),
	loc='upper left',
	borderaxespad=0.,
	frameon=False
)
fig.subplots_adjust(right=0.7)

# save figure
plt.savefig(__file__.replace(".py",".png"))
