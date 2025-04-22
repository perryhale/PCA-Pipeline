import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


### setup

# start timer
T0 = time.time()

# arguments
N_CPU = 8
DATA_PATH = './data'
DATA_RES = 224
BATCH_SIZE = 512
BATCH_TRUNC = 2


### functions

# type: () ->
def dataset_to_numpy(dataset, num_workers=0, batch_size=BATCH_SIZE, batch_trunc=BATCH_TRUNC):
	loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False if batch_trunc is None else True)
	data_x = []
	data_y = []
	for _, (x, y) in zip(range(len(loader) if batch_trunc is None else batch_trunc), loader):
		data_x.append(x.numpy())
		data_y.append(y.numpy())
	data_x = np.concatenate(data_x, axis=0)
	data_y = np.concatenate(data_y, axis=0)
	return data_x, data_y

# type: () ->
def process_dataset(key, dataset):
	
	# convert to numpy
	data_x, data_y = dataset_to_numpy(dataset)
	print(f'Converted \"{key}\" to numpy')
	print(' -> data_x', data_x.shape)
	print(' -> data_y', data_y.shape)
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')
	
	# flatten and standardize
	data_x = data_x.reshape(data_x.shape[0], -1)
	data_x = StandardScaler().fit_transform(data_x)
	print(f'Flattened and standardized \"{key}\"')
	print(' -> data_x', data_x.shape)
	print(' -> data_y', data_y.shape)
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
		'n_components':pca.n_components_,
		'mean':pca.mean_,
	}
	np.savez(f'pca_{key}.npz', **results)
	print(f'Saved results for \"{key}\"')
	print(f'[Elapsed time: {time.time() - T0:.2f}s]')


### main

# define normalising transform
transform = transforms.Compose([
	transforms.Lambda(lambda x: x.convert("RGB")),
	transforms.Resize((DATA_RES, DATA_RES)),
	transforms.ToTensor()
])

# load datasets
datasets = {
	'CIFAR10':torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform),
	'Caltech101':torchvision.datasets.Caltech101(root=DATA_PATH, download=True, transform=transform),
	'Flowers102':torchvision.datasets.Flowers102(root=DATA_PATH, download=True, transform=transform),
	'Imagenette':torchvision.datasets.Imagenette(root=DATA_PATH, download=True, transform=transform),
	'Food101':torchvision.datasets.Food101(root=DATA_PATH, download=True, transform=transform),
	'GTSRB':torchvision.datasets.GTSRB(root=DATA_PATH, download=True, transform=transform),
	'DTD':torchvision.datasets.DTD(root=DATA_PATH, download=True, transform=transform),
	'PCAM':torchvision.datasets.PCAM(root=DATA_PATH, download=True, transform=transform),
	'SEMEION':torchvision.datasets.SEMEION(root=DATA_PATH, download=True, transform=transform),
	'EuroSAT':torchvision.datasets.EuroSAT(root=DATA_PATH, download=True, transform=transform),
	'ScrewSet(1)':torchvision.datasets.ImageFolder(root='data/screwset_clean_1', transform=transform),
###!'FakeData':torchvision.datasets.FakeData(size=BATCH_TRUNC*BATCH_SIZE, image_size=(3, DATA_RES, DATA_RES), transform=transform, random_offset=time.time()), # disabled for plotting
}
print('Loaded and rescaled data')
for key, dataset in datasets.items():
	print(f' - {key} {len(dataset)} x {dataset[0][0].numpy().shape}')
print(f'[Elapsed time: {time.time() - T0:.2f}s]')

# process datasets
n_jobs = min(len(datasets), N_CPU)
print(f'Using {n_jobs} CPUs')
Parallel(n_jobs=n_jobs)([delayed(process_dataset)(key, dataset) for key, dataset in datasets.items()])
