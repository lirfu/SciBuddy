import os
import math

import numpy as np
import torch
import matplotlib.pyplot as plt


class PlotContext:
	def __init__(self, filename=None, show=False, fullscreen=True, clear=True, **kwargs):
		'''
			Manages a context for a single plot. On enter, creates a global figure with given kwargs.

			Parameters
			----------
			filename : str, optional
				File path and extension. If `ex` is defined, use it to prepend experiment path. Skipped if None. Default: None.
			show : bool, optional
				Show the pyplot plot upon exit. Default: False.
			clear : bool, optional
				Clears the figure and closes plot upon exit. Default: True.
		'''
		self.filename = filename
		self.show = show
		self.fullscreen = fullscreen
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
		if self.show and self.fullscreen:
			plt_set_fullscreen()

	def __exit__(self, type, value, trace):
		if self.filename is not None:
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

def resolve_grid_shape(N, aspect=1, force_rows=None):
	"""
		Resolves grid shape based on constraints of aspect ratio and dimension forcing.

		Parameters
		----------
		N : int
			Number of cells.
		aspect : float, optional
			Aspect ratio (width/height). Default: 1
		force_rows : int, optional
			Fixed number of rows in the grid with width filled left-to-right. If `None` is ignored. Default: `None`
	"""
	if force_rows is None:
		W = max(round(math.sqrt(N) * aspect), 1)
		H = max(math.ceil(N / W), 1)
	else:
		H = force_rows
		W = max(math.ceil(N / H), 1)
	if N < W:
		W = N
		H = 1
	return W, H

def mask_image(img, mask, color=[0,1,0], alpha=0.5):
	if isinstance(mask, np.ndarray):
		mask = mask.squeeze()[..., None] * alpha
		color = np.array(color).reshape(1,1,3)
	elif isinstance(mask, torch.Tensor):
		mask = mask.squeeze().unsqueeze(2) * alpha
		color = torch.tensor(color).reshape(1,1,3)
	else:
		RuntimeError('Unknown mask type:', type(mask))
	return (1-mask) * img + mask * color

def plt_set_fullscreen():
	backend = str(plt.get_backend())
	mgr = plt.get_current_fig_manager()
	if backend == 'TkAgg':
		if os.name == 'nt':
			mgr.window.state('zoomed')
		else:
			mgr.resize(*mgr.window.maxsize())
	elif backend == 'wxAgg':
		mgr.frame.Maximize(True)
	elif backend == 'Qt5Agg' or backend == 'Qt4Agg' or backend == 'QtAgg':
		mgr.window.showMaximized()
	else:
		print('--> Cannot maximize pyplot wintow, unknown backend: ', backend)

class PlotImageGridContext:
	def __init__(self, num_of_images, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, title=None, force_rows=None):
		self.margins = margins
		self.quiet = quiet
		self.aspect = aspect
		self.colorbar = colorbar
		self.ticks = ticks
		self.title = title
		self.N = num_of_images
		self.force_rows = force_rows

	def __enter__(self):
		self.i = 0
		self.W, self.H = resolve_grid_shape(self.N, self.aspect, self.force_rows)
		return self

	def add_image(self, img, name=None, **kwargs):
		if img is not None:
			if not self.quiet and (isinstance(img, np.ndarray) or isinstance(img, torch.Tensor)):
				if name is None:
					print(f'Image {self.i+1} of shape {img.shape} has range: [{img.min()},{img.max()}]')
				else:
					print(f'{name} of shape {img.shape} has range: [{img.min()},{img.max()}]')
			plt.subplot(self.H, self.W, self.i+1)
			if name is not None:
				plt.title(name)
			kwargs.setdefault('cmap', 'gray')
			im = plt.imshow(img, **kwargs)
			plt.gca().axes.xaxis.set_visible(self.ticks)
			plt.gca().axes.yaxis.set_visible(self.ticks)
			if self.colorbar:
				plt.colorbar(im)
		self.i += 1

	def __exit__(self, type, value, trace):
		plt.gcf().subplots_adjust(
			top=1-self.margins,
			bottom=self.margins,
			left=self.margins,
			right=1-self.margins,
			hspace=self.margins,
			wspace=self.margins
		)
		if self.title is not None:
			plt.suptitle(self.title)

def show_images(*imgs, names=None, fullscreen=True, **kwargs):
	with PlotContext(show=True, fullscreen=fullscreen):
		draw_images(*imgs, names=names, **kwargs)

def draw_images(*imgs, names=None, margins=0.01, quiet=True, aspect=16/9, colorbar=False, ticks=False, title=None, force_rows=None, **kwargs):
	with PlotImageGridContext(len(imgs), margins, quiet, aspect, colorbar, ticks, title, force_rows) as pc:
		for i,img in enumerate(imgs):
			if names is not None:
				n = names[i]
			else:
				n = None
			pc.add_image(img, n, **kwargs)


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
