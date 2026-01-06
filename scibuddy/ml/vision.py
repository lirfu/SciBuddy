import math
from typing import Callable

import torch
import torchvision
from tqdm import tqdm

from ..io.image import chw_to_hwc


class ConvolutionKernel(torch.nn.Module):
	"""
		Simple kernel convolution module.
	"""
	def __init__(self, kernel, padding=None, padding_mode='constant', padding_value=0, stride=1, dilation=1, convolution_fn=torch.nn.functional.conv2d):
		super(ConvolutionKernel, self).__init__()
		self.kernel = kernel
		self.convolver = convolution_fn
		# Expand kernel shape.
		if len(self.kernel.shape) == 2:
			self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
		elif len(kernel.shape) == 3:
			self.kernel = self.kernel.unsqueeze(0)
		else:
			raise RuntimeError('Valid kernel must have 2 or 3 dimensions.')
		# Calculate padding if not given.
		if padding is None:  # Estimate the 'same' padding (for even kernels padded right/bottom, for odd kernels on both sides).
			H, W = self.kernel.shape[-2], self.kernel.shape[-1]
			padding = (
				int(H/2) - 1 + H%2,
				int(H/2),
				int(W/2) - 1 + W%2,
				int(W/2)
			)
		self.padding = padding
		self.padding_mode = padding_mode
		self.padding_value = padding_value
		self.stride = stride
		self.dilation = dilation

	def forward(self, img):
		kernel = self.kernel
		if len(img.shape) == 2:  # Add missing channels dimension
			img = img.unsqueeze(0)
		if len(img.shape) == 3:  # Add missing batch dimension.
			img = img.unsqueeze(0)
		if img.shape[1] != kernel.shape[1]:  # Apply same kernel across multiple channels.
			kernel = kernel.expand(-1,img.shape[1],-1,-1)
		if img.shape[0] != kernel.shape[0]:  # Apply same kernel across multiple channels.
			kernel = kernel.expand(img.shape[0],-1,-1,-1)
		img = torch.nn.functional.pad(
			img,
			pad=self.padding,
			mode=self.padding_mode,
			value=self.padding_value
		)
		return self.convolver(  # FIXME: Supports only 'float' types in torch<=1.10.0
			img.float(),
			kernel,
			stride=self.stride,
			dilation=self.dilation
		)

	def __repr__(self):
		return str(self.kernel)

class MultistageConvolutionKernel(torch.nn.Module):
	"""
		Kernel convolution module for multi-stage convolutions (e.g. vertical kernel following horizontal kernel). If given kernel is not instance of `ConvolutionKernel`, constructs one with given kernel array.
	"""
	def __init__(self, kernels, inter_process=None):
		super(MultistageConvolutionKernel, self).__init__()
		if not isinstance(kernels, list):
			kernels = [kernels]
		layers = []
		for k in kernels:
			if isinstance(k, ConvolutionKernel):
				layers.append(k)
			else:
				layers.append(ConvolutionKernel(k))
		self.kernels = layers
		self.inter_process = inter_process

	def forward(self, img):
		res = None
		for k in self.kernels:
			inter = k(img)
			if self.inter_process is not None:
				inter = self.inter_process(inter)
			if res is None:
				res = inter
			else:
				res += inter
		return res

class GradientKernel(torch.nn.Module):
	"""
		Kernel convolution module for image gradients from decomposed convolutions (e.g. vertical kernel and horizontal kernel). If given kernel is not instance of `ConvolutionKernel`, constructs one with given kernel array.
		
		Parameters
		----------
		kernel_v, kernel_h: Union(ConvolutionKernel,torch.Tensor)
			Vertical and horizontal gradient kernel.
		length: bool
			If `True` returns only the length of the gradient vector. Otherwise, returns horizontal and vertical component of gradient as two-channel image.
		normalize: bool
			If `True` normalizes gradient vector. Active only if returning graident vector.
	"""
	def __init__(self, kernel_h, kernel_v, length=False, normalize=False, **kwargs):
		super(GradientKernel, self).__init__()
		if not isinstance(kernel_h, ConvolutionKernel):
			kernel_h = ConvolutionKernel(kernel_h)
		if not isinstance(kernel_v, ConvolutionKernel):
			kernel_v = ConvolutionKernel(kernel_v)
		self.kh = kernel_h
		self.kv = kernel_v
		self.length = length
		self.normalize = normalize

	def forward(self, img):
		v, h = self.kv(img), self.kh(img)
		if self.length:
			return torch.sqrt(v**2 + h**2)
		g = torch.cat([h,v], dim=1)
		if self.normalize:
			return g / torch.norm(g, dim=1)
		return g

class IdentityKernel(ConvolutionKernel):
	def __init__(self, **kwargs):
		super(IdentityKernel, self).__init__(torch.tensor([[1.]]), **kwargs)

class SobelKernel(MultistageConvolutionKernel):
	"""
		Sobel's corner detector.
	"""
	def __init__(self):
		super(SobelKernel, self).__init__([
			torch.FloatTensor([
				[-1, 0, 1],
				[-2, 0, 2],
				[-1, 0, 1]
			]) / 4,
			torch.FloatTensor([
				[-1, -2, -1],
				[ 0,  0,  0],
				[ 1,  2,  1]
			]) / 4
		], inter_process=lambda x: x.pow(2))

	def forward(self, img):
		return super().forward(img).sqrt()

class SobelGradientKernel(GradientKernel):
	"""
		Sobel's gradient kernel.
	"""
	def __init__(self, **kwargs):
		super(SobelGradientKernel, self).__init__(
			torch.FloatTensor([
				[-1, 0, 1],
				[-2, 0, 2],
				[-1, 0, 1]
			]) / 4.,
			torch.FloatTensor([
				[-1, -2, -1],
				[ 0,  0,  0],
				[ 1,  2,  1]
			]) / 4
		, **kwargs)

class RobertsKernel(MultistageConvolutionKernel):
	"""
		Robert's edge detector.
	"""
	def __init__(self):
		super(RobertsKernel, self).__init__([
			torch.FloatTensor([
				[1,  0],
				[0, -1]
			]) / 2,
			torch.FloatTensor([
				[0, 1],
				[-1, 0]
			]) / 2
		], inter_process=lambda x: x**2)
	
	def forward(self, img):
		return super().forward(img).sqrt()

class RobertsGradientKernel(GradientKernel):
	"""
		Robert's gradient kernel.
	"""
	def __init__(self, **kwargs):
		super(RobertsGradientKernel, self).__init__(
			torch.FloatTensor([
				[1,  0],
				[0, -1]
			]),
			torch.FloatTensor([
				[0, 1],
				[-1, 0]
			])
		, **kwargs)

	def forward(self, img):
		return super().forward(img) / 2

class BoxKernel(ConvolutionKernel):
	"""
		Box averaging kernel for blurring.
	"""
	def __init__(self, size, normalize=True, **kwargs):
		super(BoxKernel, self).__init__(
			torch.ones(size,size, dtype=torch.float) / (size**2 if normalize else 1.),
			**kwargs
		)

	def forward(self, img):
		return super().forward(img)

class ErosionKernel(BoxKernel):
	"""
		Binary erosion kernel (like Box), expects a [0,1] normalized image. Performs additional binarization by clearing values smaller than given step.
	"""
	def __init__(self, size, binarize=True, binarization_step=1., **kwargs):
		super(ErosionKernel, self).__init__(size, padding_value=1, **kwargs)
		self.bin = binarize
		self.step = binarization_step

	def forward(self, img):
		img = super().forward(img)
		if self.bin:
			return img >= self.step
		return img

class DilationKernel(BoxKernel):
	"""
		Binary dilation kernel (like Box), expects a [0,1] normalized image. Performs additional binarization by setting values larger than given step.
	"""
	def __init__(self, size, binarize=True, binarization_step=0., **kwargs):
		super(DilationKernel, self).__init__(size, padding_value=0, **kwargs)
		self.bin = binarize
		self.step = binarization_step

	def forward(self, img):
		img = super().forward(img)
		if self.bin:
			return img > self.step
		return img

class HorizontalDilation(ConvolutionKernel):
	def __init__(self, size, **kwargs):
		k_size = size * 2 + 1
		super(HorizontalDilation, self).__init__(torch.cat([
			torch.zeros((k_size, size)), torch.ones((k_size, 1)), torch.zeros((k_size, size))
		], dim=1).to(torch.float32).T, convolution_fn=maxconv, **kwargs)

class VerticalDilation(ConvolutionKernel):
	def __init__(self, size, **kwargs):
		k_size = size * 2 + 1
		super(VerticalDilation, self).__init__(torch.cat([
			torch.zeros((k_size, size)), torch.ones((k_size, 1)), torch.zeros((k_size, size))
		], dim=1).to(torch.float32), convolution_fn=maxconv, **kwargs)

class GaussianKernel(ConvolutionKernel):
	def __init__(self, half_size, std, **kwargs):  # TODO Dynammic kernel size based on std?
		k = torch.arange(1, 2*half_size+2)
		gx, gy = torch.meshgrid(k, k)
		k = torch.exp( -0.5 / std**2 * ((gx-half_size-1)**2+(gy-half_size-1)**2) )
		k /= k.sum()
		super(GaussianKernel, self).__init__(k, **kwargs)

def custom_conv(img: torch.Tensor, kernel: torch.Tensor, fn: Callable, stride: int=1, batching=None):
	"""
		Performs convolution with custom join function.

		Parameters
		----------
		img: torch.Tensor
			Image to process [N,C,H,W].
		kernel: torch.Tensor
			Kernel to apply over image [O,C,KH,KW].
		fn: function
			Function for calculating result. Gets crops of input image [N,1,C,S,KH,KW] and kernel [1,O,C,1,KH,KW], outputs accumulated (over C, KH and KW) result [N,O,S].

		Returns
		----------
		img: torch.Tensor
			Resulting image [N,O,H-KH+1,W-KW+1].
	"""
	N, C, H, W = img.shape
	O, _, KH, KW = kernel.shape

	if batching is None:
		features = img.unfold(2,KH,stride).unfold(3,KW,stride).flatten(2,3)
		img2 = fn(features.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(3))
		return img2.reshape((N, O, H-KH+1 ,W-KW+1))
	elif batching == 'width':
		patches = img.unfold(3,KW,stride)
		img2 = torch.zeros(N, O, int((H-KH)/stride+1), int((W-KW)/stride+1))
		for i in tqdm(range(patches.shape[3]), desc='Processing custom convolution', leave=False):
			f = patches[:,:,:,i,:]
			features = f.unfold(2,KH,stride)
			res = fn(features.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(3))
			img2[:, :, :, i] = res
		return img2
	elif batching == 'height':
		patches = img.unfold(2,KH,stride)
		img2 = torch.zeros(N, O, int((H-KH)/stride+1), int((W-KW)/stride+1))
		for i in tqdm(range(patches.shape[2]), desc='Processing custom convolution', leave=False):
			f = patches[:,:,i,:,:]
			features = f.unfold(2,KW,stride)
			res = fn(features.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(3))
			img2[:, :, i, :] = res
		return img2
	else:
		raise RuntimeError('Unknown batching type', batching)
	# # Manual batching.
	# img2 = torch.empty(N, O, H, W)
	# def iterator():
	# 	for i in range(int((H-KH)/stride+1)):
	# 		for j in range(int((W-KW)/stride+1)):
	# 			yield i*stride, j*stride
	# 	yield None, None
	# batch = torch.empty(N,C,batch_size,KH,KW)
	# idi, idj = torch.meshgrid(torch.arange(KH), torch.arange(KW), indexing='ij')
	# it = iterator()
	# run = True
	# pbar = tqdm(desc='Running batched convolution')
	# pbar.reset(total=H*W)
	# while run:
	# 	indices = []
	# 	for b in tqdm(range(batch_size), desc='Batching...', leave=False):
	# 		i, j = next(it)
	# 		if i is None:
	# 			run = False
	# 			break
	# 		batch[...,b,:,:] = img[:, :, i+idi, i+idj]
	# 		# batch[...,b,:,:] = img[:, :, i:i+KH, j:j+KW]
	# 		indices.append([i,j])
	# 	indices = torch.tensor(indices)
	# 	feat = fn(batch.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(3))
	# 	pbar.update(len(indices))
	# 	for b, ij in enumerate(indices):
	# 		img2[:, :, ij[0], ij[1]] = feat[...,b]
	# pbar.close()
	# return img2

def maxconv(img, kernel, stride=1, **kwargs):
	return custom_conv(img, kernel, lambda features, kernel: (features*kernel).max(dim=-1).values.max(dim=-1).values, stride)

class VerticalEdgeKernel(ConvolutionKernel):
	def __init__(self, **kwargs):
		super(VerticalEdgeKernel, self).__init__(torch.tensor([
			[-1,0,1],
			[-1,0,1],
			[-1,0,1]
		], dtype=torch.float32) / 3, **kwargs)

class HorizontalEdgeKernel(ConvolutionKernel):
	def __init__(self, **kwargs):
		super(HorizontalEdgeKernel, self).__init__(torch.tensor([
			[-1,-1,-1],
			[0,0,0],
			[1,1,1]
		], dtype=torch.float32) / 3, **kwargs)

def rotated_kernel(kernel: ConvolutionKernel, angle: float, binarize: bool=False):
	if isinstance(kernel, ConvolutionKernel):
		k = kernel.kernel
	else:
		k = kernel
		if len(k.shape) == 2:
			k.unsqueeze(0)
		elif len(k.shape) == 3:
			k.unsqueeze(0).unsqueeze(0)
	kk = torchvision.transforms.functional.rotate(
		k,
		math.degrees(angle),
		interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
		expand=False
	)
	if binarize:
		kk[kk>0] = 1.
	kernel.kernel = kk
	return kernel

class BlurEdgeDetector(torch.nn.Module):
	"""
		Detects edges in image by comparing it with blurred version.
	"""
	def __init__(self, blurrer, highlight_inside=True, highlight_outside=True):
		super(BlurEdgeDetector, self).__init__()
		self.blurrer = blurrer
		self.inside = highlight_inside
		self.outside = highlight_outside

	def forward(self, img):
		img_b = self.blurrer(img)
		if self.inside and self.outside:
			return img + img_b - 2. * img * img_b
		elif self.inside:
			return img * (1-img_b)
		elif self.outside:
			return img_b * (1-img)
		else: 
			return 1 - (img + img_b - 2. * img * img_b)

# VFX.

class CumulativeImageGhosting:
	def __init__(self, size=10):
		self.size = size
		self.buffer = None
		self.image = None

	def __call__(self, img):
		if self.image is not None:
			l = len(self.buffer)
			if l < self.size:
				self.image = self.image + img
			else:
				img_l = self.buffer.pop(0)
				self.image = self.image - img_l + img
			self.buffer.append(img)
			return self.image / (l+1)
		else:
			self.image = img
			self.buffer = [img]
			return self.image

	def get_image(self):
		if self.image is not None and self.buffer is not None:
			return self.image / len(self.buffer)
		return None

	def reset(self):
		self.buffer = None
		self.image = None


if __name__ == '__main__':
	torch.manual_seed(42)
	# Display kernel functions.
	from scibuddy.io.pyplot import show_images
	size = 16
	kernels = {
		'Identity': IdentityKernel(),

		'SobelGradient': SobelGradientKernel(length=False),
		'SobelGradient1': SobelGradientKernel(length=True),
		'Monster': torch.nn.Sequential(SobelGradientKernel(length=False), BoxKernel(2)),
		'RobertsGradient': RobertsGradientKernel(length=False),

		'Sobel': SobelKernel(),
		'Roberts': RobertsKernel(),
		'Box': BoxKernel(3),
		'Erosion': ErosionKernel(3),
		'Dilation': DilationKernel(3),
		'Gaussian': GaussianKernel(3, 1),

		'Vertical edges': VerticalEdgeKernel(),
		'Rotation -15': rotated_kernel(VerticalEdgeKernel(), -math.pi / 12),
		'Rotation -30': rotated_kernel(VerticalEdgeKernel(), -math.pi / 6),
		'Rotation -45': rotated_kernel(VerticalEdgeKernel(), -math.pi / 4),
		'Rotation -60': rotated_kernel(VerticalEdgeKernel(), -math.pi / 3),
		'Rotation -75': rotated_kernel(VerticalEdgeKernel(), -math.pi * 5 / 12),
		'Rotation -90': rotated_kernel(VerticalEdgeKernel(), -math.pi / 2),
		'Horizontal edges': HorizontalEdgeKernel(),

		'Vertical dilation 1': VerticalDilation(1),
		'Vertical dilation 2': VerticalDilation(2),
		'Vertical dilation 3': VerticalDilation(3),
		'Vertical dilation 4': VerticalDilation(4),
		'Vertical dilation 5': VerticalDilation(5),
		'Vertical dilation 6': VerticalDilation(6),
		'Vertical dilation 7': VerticalDilation(7),

		'Vertical dilation': VerticalDilation(size),
		'Rotation -15': rotated_kernel(VerticalDilation(size), -math.pi / 12, True),
		'Rotation -30': rotated_kernel(VerticalDilation(size), -math.pi / 6, True),
		'Rotation -45': rotated_kernel(VerticalDilation(size), -math.pi / 4, True),
		'Rotation -60': rotated_kernel(VerticalDilation(size), -math.pi / 3, True),
		'Rotation -75': rotated_kernel(VerticalDilation(size), -math.pi * 5 / 12, True),
		'Rotation -90': rotated_kernel(VerticalDilation(size), -math.pi / 2, True),
		'Horizontal dilation': HorizontalDilation(size)

		# 'Test1': ConvolutionKernel(torch.tensor([
		# 	[1,1],
		# 	[0,0]
		# ], dtype=torch.float32)/2, convolution_fn=maxconv)
	}
	img = torch.zeros((32,32), dtype=torch.float32)
	img[0,:] = 1
	img[:6,0] = 1
	img[:6,-1] = 1
	img[10:21, 5:11] = 1
	img[8, 20] = 1
	img[20, 15:26] = 1
	img[15:26, 20] = 1
	for i in range(7):
		img[-i, i] = 1
		img[-i+1, i] = 1
		img[6-i, i] = 1
	# img *= torch.rand_like(img)
	# img = torchvision.transforms.Resize((1024)*2)(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
	
	# Construct and display results.
	images = []
	for n,k in kernels.items():
		r = k(img).squeeze(0)
		if r.shape[0] == 2: # Stack in RB channels.
			# print(f'Range for {n}: [{r.min()}, {r.max()}]  {r[:,15,4]}  {r[:,9,7]}  {r[:,20,7]}  {r[:,15,10]}')
			r = (r - r.min()) / (r.max() - r.min())  # Normalize to [0,1].
			r = rb_to_rgb(r)  # FIXME
		images.append(chw_to_hwc(r).numpy())
	show_images(*images, names=list(kernels.keys()), quiet=False)

	# # Live camera demo.
	# from camera_utils import *
	# import cv2, time
	# cam = ComputerCamera()
	# k = SobelKernel()
	# def get_img(cam, k):
	# 	img = cvframe_to_torch_float(cam.get_frame())
	# 	img = k(img).squeeze(0)
	# 	return torch_float_to_cvframe(img)
	# cv2.namedWindow("preview")
	# frame = get_img(cam, k)
	# t = time.time()
	# while frame is not None:
	# 	cv2.imshow("preview", frame)
	# 	key = cv2.waitKey(1)
	# 	if key == 27: # exit on ESC
	# 		break
	# 	dt = time.time()-t
	# 	t = time.time()
	# 	print('{: 8.2f} ms ({: 5.2f} FPS)'.format(dt*1000., 1./dt))
	# 	frame = get_img(cam, k)
	# cv2.destroyWindow("preview")