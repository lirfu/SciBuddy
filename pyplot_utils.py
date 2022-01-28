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
			ex : Experiment, optional
				Experiment used to save the image upon exit. Skipped if None. Default: None.
			filename : str, optional
				File path and extension. If `ex` is defined, use it to prepend experiment path. Skipped if None. Default: None.
			show : bool, optional
				Show the pyplot plot upon exit. Default: False.
			clear : bool, optional
				Clears the figure and closes plot upon exit. Default: True.
		'''
		self.ex = ex
		self.filename = filename
		self.show = show
		self.clear = clear
		self.kwargs = kwargs

	@staticmethod
	def clear():
		plt.cla()
		plt.clf()
		plt.close()

	def __enter__(self):
		PlotContext.clear()
		plt.figure(clear=True, **self.kwargs)

	def __exit__(self, type, value, trace):
		if self.filename is not None:
			if self.ex is not None:
				self.ex.save_pyplot_image(self.filename)
			else:
				plt.savefig(self.filename, bbox_inches='tight', pad_inches=0, transparent=False, dpi=self.kwargs.get('dpi', None))
		if self.show:
			plt.show()
		if self.clear:
			PlotContext.clear()

def draw_loss_curve(losses, label=None, xlabel='Iterations', ylabel='Loss'):
	'''
		Plot the loss curve from an array of losses. Shows the grid.
	'''
	if len(losses[0]) == 1:
		plt.plot(losses, label=label)
	else:
		plt.plot([l[0] for l in losses], [l[1] for l in losses], label=label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(True)

def draw_class_distribution(features, labels, colormap='hsv', marker='.', sizes=20, legend=True):
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

def draw_images_grid(images, grid, labels=None, labels_prefix='', fontsize=10):
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

def draw_ranges(x, y, y_m, y_M, linecolor='black', rangestyle=':', rangecolor='blue', rangealpha=0.25):
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

def mask_image(img, mask, color=[0,1,0], alpha=0.5):
	if isinstance(mask, np.ndarray):
		mask = mask.squeeze()[..., None] * alpha
		color = np.array(color).reshape(1,1,3)
	if isinstance(mask, torch.Tensor):
		mask = mask.squeeze().unsqueeze(2) * alpha
		color = torch.tensor(color).reshape(1,1,3)
	else:
		RuntimeError('Unknown mask type:', type(mask))
	return (1-mask) * img + mask * color

def show_images(*imgs, names=None, **kwargs):
	with PlotContext(show=True):
		draw_images(*imgs, names=names, **kwargs)

def draw_images(*imgs, names=None, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, **kwargs):
	if isinstance(names, str):
		names = [names]
	N = len(imgs)
	Nr = math.sqrt(N)
	W = max(round(Nr * aspect), 1)
	H = max(math.ceil(N / W), 1)
	for i,img in enumerate(imgs):
		if not quiet and (isinstance(img, np.ndarray) or isinstance(img, torch.Tensor)):
			if names is None:
				print(f'Image {i+1} of shape {img.shape} has range: [{img.min()},{img.max()}]')
			else:
				print(f'{names[i]} of shape {img.shape} has range: [{img.min()},{img.max()}]')
		plt.subplot(H,W,i+1)
		if names is not None:
			plt.title(names[i])
		im = plt.imshow(img, cmap=kwargs.get('cmap', 'gray'), **kwargs)
		plt.gca().axes.xaxis.set_visible(ticks)
		plt.gca().axes.yaxis.set_visible(ticks)
		if colorbar:
			plt.colorbar(im)
	plt.gcf().subplots_adjust(
		top=1-margins,
		bottom=margins,
		left=margins,
		right=1-margins,
		hspace=margins,
		wspace=margins
	)

def draw_precision_recall_curve(p, r, class_names=None, title=None, grid=True, padding=0.01):
	C = 1
	if len(p.shape) >= 1:
		C = p.shape[1]
	if class_names is None:
		class_names = [f'Class {c+1}' for c in range(C)]
	for c, name in enumerate(class_names):
		plt.step(r[:,c], p[:,c], where='post', marker='o', markersize=2, label=name)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
	plt.legend()
	plt.grid(grid)
	plt.xlim(-padding,1+padding)
	plt.ylim(-padding,1+padding)
	if title is not None:
		plt.title(title)
