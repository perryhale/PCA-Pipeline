import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# start timer
T0 = time.time()

# define transform
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor()
])

# load datasets
datasets = {
	'cifar10':		torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
#	'caltech101':	torchvision.datasets.Caltech101(root='./data', download=True, transform=transform),
#	'flowers102':	torchvision.datasets.Flowers102(root='./data', download=True, transform=transform),
#	'imagenette':	torchvision.datasets.Imagenette(root='./data', download=True, transform=transform),
#	'food101':		torchvision.datasets.Food101(root='./data', download=True, transform=transform),
#	'gtsrb':		torchvision.datasets.GTSRB(root='./data', download=True, transform=transform),
}
print('Loaded/Scaled data')
for key in datasets: print(key)
print(f'[Elapsed time: {time.time() - T0:.2f}s]')

# process datasets
for key, dataset in datasets.items():
	
	# convert to numpy
	data_x = []
	data_y = []
	for blah, (x, y) in zip(range(2), dataset):
		x_np = x.numpy()
		data_x.append(x_np)
		data_y.append(y)
	data_x = np.stack(data_x)
	data_y = np.array(data_y)
	print("data_x:", data_x.shape)
	print("data_y:", data_y.shape)
	print(f'Converted \"{key}\" to numpy')
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')
	
	# flatten and standardize
	data_x = data_x.reshape(data_x.shape[0], -1)
	data_x = StandardScaler().fit_transform(data_x)
	print("data_x:", data_x.shape)
	print("data_y:", data_y.shape)
	print(f'Flattened and standardized \"{key}\"')
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')
	
	# perform pca
	pca = PCA()
	pca.fit(data_x)
	print(f'Done PCA for \"{key}\"')
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')
	
	# record results
	results = {
		'components':pca.components_,
		'explained_variance':pca.explained_variance_,
		'explained_variance_ratio':pca.explained_variance_ratio_,
		'singular_values':pca.singular_values_,
		'mean':pca.mean_,
		'n_components':pca.n_components_,
		'mean':pca.mean_,
	}
	np.savez(f'pca_{key}.npz', **results)
	print(f'Saved results for \"{key}\"')
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')
