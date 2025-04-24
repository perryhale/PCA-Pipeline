import numpy as np
import matplotlib.pyplot as plt
import glob

# define paths
file_paths = glob.glob('*.npz')

# define colors
colors = ['grey', 'lightsteelblue']

# determine AUC scores
areas = {}
for path in file_paths:
	key = path.replace('pca_', '').replace('.npz', '')
	data = np.load(path)
	ratio = data['explained_variance_ratio']
	area = np.trapezoid(np.cumsum(ratio))
	areas[key] = area

# sort areas
sorted_areas = {k: v for k, v in sorted(areas.items(), key=lambda item: item[1], reverse=True)}

# init plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(sorted_areas.keys(), sorted_areas.values(), color=colors[:len(sorted_areas)])

# add labels
ax.set_ylabel('Cumulative Explained Variance Ratio AUC')
ax.set_xticks(np.arange(len(sorted_areas.keys())))
ax.set_xticklabels(sorted_areas.keys(), rotation=45, ha='right')
for bar in bars:
	yval = bar.get_height()
	ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')

# finalise
ax.grid(True, axis='y')
plt.tight_layout()

# save figure
plt.savefig(__file__.replace('.py','.png'))
