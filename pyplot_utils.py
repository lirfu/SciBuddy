import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class PlotContext:
	def __init__(self, ex=None, filename=None, show=False, clear=True, **kwargs):
		'''
			Manages a context for a single plot. On enter, creates a global figure with given kwargs.

			Parameters
			----------
			ex : Experiment
				Experiment used to save the image upon exit. Skipped if None. Default: None.
			filename : str
				File path and extension type relative to the given experiment path upon exit. Skipped if None. Default: None.
			show : bool
				Show the pyplot plot upon exit. Default: False.
			clear : bool
				Clears the figure and closes plot upon exit. Default: True.
		'''
		self.ex = ex
		self.filename = filename
		self.show = show
		self.clear = clear
		self.kwargs = kwargs

	def __enter__(self):
		plt.figure(clear=True, **self.kwargs)

	def __exit__(self, type, value, trace):
		if self.ex is not None and self.filename is not None:
			self.ex.save_pyplot_image(self.filename)
		if self.show:
			plt.show()
		if self.clear:
			plt.clf()
			plt.close()

def plot_loss_curve(losses, xlabel='Iterations', ylabel='Loss'):
	'''
		Plot the loss curve from an array of losses. Shows the grid.
	'''
	plt.plot(losses)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(True)

def plot_class_distribution(features, labels, colormap='hsv', marker='.', sizes=20, legend=True):
	'''
		Scatters the 2D points with assigned labels and adds a label legend.
	'''
	edg = 'none'
	if len(marker) > 1:
		edg = 'black'
		marker = marker[0]
	scatter = plt.scatter(features[:,0], features[:,1], marker=marker, c=labels, edgecolors=edg, cmap=plt.get_cmap(colormap), s=sizes)
	if legend:
		legend = plt.legend(*scatter.legend_elements(), title='Labels')
		plt.gca().add_artist(legend)

def plot_images_grid(images, grid, labels=None, labels_prefix='', fontsize=10):
	'''
		Draws given 1D list of images into a grid, filling row-by-row (pyplot style).
	'''
	for i,img in enumerate(tqdm(images, leave=False, desc='Drawing image grid'), 1):
		ax = plt.subplot(*grid, i)
		ax.imshow(img, cmap='gray', interpolation='none')
		ax.set_xticks([])
		ax.set_yticks([])
		if labels is not None:
  			ax.set_title(labels_prefix + str(labels[i-1]), fontsize=fontsize)

def plot_ranges(x, y, y_m, y_M, linecolor='black', rangestyle=':', rangecolor='blue', rangealpha=0.25):
	ax = plt.gca()
	ax.plot(x, y, color=linecolor)
	ax.plot(x, y_m, color=linecolor, linestyle=rangestyle)
	ax.plot(x, y_M, color=linecolor, linestyle=rangestyle)
	ax.fill_between(x, y, y_M, facecolor=rangecolor, alpha=rangealpha)
	ax.fill_between(x, y_m, y, facecolor=rangecolor, alpha=rangealpha)

def make_2d_grid(m, M, shape):
	"""
		Make a 2D point grid from linspaces across dimensions.

		Parameters
		----------
		m :	float or Tuple(float)
			Minimum value/s for linspace.
		M :	float or Tuple(float)
			Maximum value/s for linspace.
		shape :	Tuple(int,int)
			Shape of grid following the row-major order (H,W).
		
		Returns
		----------
		torch.FloatTensor(H,W,2)
			Grid of 2D coordinates.
	"""
	def resolve(v):
		if (isinstance(v, tuple) or isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)):
			if len(v) > 1:
				return v[0], v[1]
			else:
				return v[0], v[0]
		else:
			return v, v
	mx, my = resolve(m)
	Mx, My = resolve(M)
	h = torch.tensor(np.linspace(my, My, shape[0])).float()
	w = torch.tensor(np.linspace(mx, Mx, shape[1])).float()
	gw, gh = torch.meshgrid(h, w)
	return torch.stack([gh, gw], dim=2)


def fit_grid_shape(length, prefer_width=True):
	dsq = math.sqrt(length)
	if prefer_width:
		shape = (math.ceil(dsq), math.floor(dsq))
		if math.modf(dsq)[0] >= 0.5:
			shape = (shape[0], shape[1]+1)
	else:
		shape = (math.floor(dsq), math.ceil(dsq))
		if math.modf(dsq)[0] >= 0.5:
			shape = (shape[0]+1, shape[1])
	return shape