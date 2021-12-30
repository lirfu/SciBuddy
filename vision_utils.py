import math
from scipy.ndimage import interpolation
import torch
import torchvision


def chw_to_hwc(img):
	return img.moveaxis(0,2)

def hwc_to_chw(img):
	return img.moveaxis(2,0)

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
			kernel = kernel.repeat(1,img.shape[1],1,1)
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
	def __init__(self, kernels):
		super(MultistageConvolutionKernel, self).__init__()
		if not isinstance(kernels, list):
			kernels = [kernels]
		layers = []
		for k in kernels:
			if isinstance(k, ConvolutionKernel):
				layers.append(k)
			else:
				layers.append(ConvolutionKernel(k))
		self.kernels = torch.nn.Sequential(*layers)

	def forward(self, img):
		return self.kernels(img)

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
	def __init__(self, kernel_v, kernel_h, length=False, normalize=False, **kwargs):
		super(GradientKernel, self).__init__()
		if not isinstance(kernel_v, ConvolutionKernel):
			kernel_v = ConvolutionKernel(kernel_v)
		if not isinstance(kernel_h, ConvolutionKernel):
			kernel_h = ConvolutionKernel(kernel_h)
		self.kv = kernel_v
		self.kh = kernel_h
		self.length = length
		self.normalize = normalize

	def forward(self, img):
		v, h = self.kv(img), self.kh(img)
		if self.length:
			return torch.sqrt(v**2 + h**2)
		g = torch.cat([h,v], dim=1)
		if self.normalize:
			return g / torch.norm(g, dim=1)

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
			]),
			torch.FloatTensor([
				[ 1,  2,  1],
				[ 0,  0,  0],
				[-1, -2, -1]
			])
		])

	def forward(self, img):
		return super().forward(img) / 9

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
			]),
			torch.FloatTensor([
				[ 1,  2,  1],
				[ 0,  0,  0],
				[-1, -2, -1]
			])
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
			]),
			torch.FloatTensor([
				[ 0, 1],
				[-1, 0]
			])
		])
	
	def forward(self, img):
		return super().forward(img) / 2

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
				[ 0, 1],
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
			torch.ones(size,size, dtype=torch.float),
			**kwargs
		)
		if normalize:
			self.norm = size*size
		else:
			self.norm = 1

	def forward(self, img):
		return super().forward(img) / self.norm

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

def maxconv(img, kernel, stride=1, **kwargs):
	kernel_h, kernel_w = kernel.shape[-1], kernel.shape[-2]
	features = img.unfold(2,kernel_h,stride).unfold(3,kernel_w,stride).flatten(2,3)
	img2 = (features.unsqueeze(1) * kernel.unsqueeze(0).unsqueeze(3)).max(dim=-1).values.max(dim=-1).values
	return img2.reshape((img.shape[0],kernel.shape[1],img.shape[2]-kernel_h+1,img.shape[3]-kernel_w+1))

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
	from tools import show_images
	size = 16
	kernels = {
		'Identity': IdentityKernel(),
		'Test1': ConvolutionKernel(torch.tensor([
			[1,1],
			[0,0]
		], dtype=torch.float32)/2, convolution_fn=maxconv)

		# 'Sobel': SobelKernel(),
		# 'Roberts': RobertsKernel(),
		# 'Box': BoxKernel(3),
		# 'Erosion': ErosionKernel(3),
		# 'Dilation': DilationKernel(3),
		# 'Gaussian': GaussianKernel(3, 1),

		# 'Vertical edges': VerticalEdgeKernel(size),
		# 'Rotation -15': RotateKernel(VerticalEdgeKernel(size), -math.pi / 12),
		# 'Rotation -30': RotateKernel(VerticalEdgeKernel(size), -math.pi / 6),
		# 'Rotation -45': RotateKernel(VerticalEdgeKernel(size), -math.pi / 4),
		# 'Rotation -60': RotateKernel(VerticalEdgeKernel(size), -math.pi / 3),
		# 'Rotation -75': RotateKernel(VerticalEdgeKernel(size), -math.pi * 5 / 12),
		# 'Rotation -90': RotateKernel(VerticalEdgeKernel(size), -math.pi / 2),
		# 'Horizontal edges': HorizontalEdgeKernel(size)

		# 'Vertical dilation 1': VerticalDilation(1),
		# 'Vertical dilation 2': VerticalDilation(2),
		# 'Vertical dilation 3': VerticalDilation(3),
		# 'Vertical dilation 4': VerticalDilation(4),
		# 'Vertical dilation 5': VerticalDilation(5),
		# 'Vertical dilation 6': VerticalDilation(6),
		# 'Vertical dilation 7': VerticalDilation(7),

		# 'Vertical dilation': VerticalDilation(size),
		# 'Rotation -15': rotated_kernel(VerticalDilation(size), -math.pi / 12, True),
		# 'Rotation -30': rotated_kernel(VerticalDilation(size), -math.pi / 6, True),
		# 'Rotation -45': rotated_kernel(VerticalDilation(size), -math.pi / 4, True),
		# 'Rotation -60': rotated_kernel(VerticalDilation(size), -math.pi / 3, True),
		# 'Rotation -75': rotated_kernel(VerticalDilation(size), -math.pi * 5 / 12, True),
		# 'Rotation -90': rotated_kernel(VerticalDilation(size), -math.pi / 2, True),
		# 'Horizontal dilation': HorizontalDilation(size)
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
	images = []
	for n,k in kernels.items():
		images.append(chw_to_hwc(k(img).squeeze(0)).numpy())
	show_images(*images, names=list(kernels.keys()))

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