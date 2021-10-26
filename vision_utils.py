import math
import torch

class ConvolveKernel(torch.nn.Module):
	"""
		Simple kernel convolution module.
	"""
	def __init__(self, kernel, padding=None, padding_mode='constant', padding_value=0, stride=1, dilation=1):
		super(ConvolveKernel, self).__init__()
		self.kernel = kernel
		if len(self.kernel.shape) == 2:
			kernel = kernel.unsqueeze(0).unsqueeze(0)
		elif len(kernel.shape) == 3:
			kernel = kernel.unsqueeze(0)
		else:
			raise RuntimeError('Valid kernel must have 2 or 3 dimensions.')
		if padding is None:
			self.kernel.shape[-1]
			padding = (  # 'Same' padding.
				int((self.kernel.shape[-2] - 1) / 2),
				int((self.kernel.shape[-2] - 1) / 2),
				int((self.kernel.shape[-1] - 1) / 2),
				int((self.kernel.shape[-1] - 1) / 2)
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
		return torch.nn.functional.conv2d(  # FIXME: Implemeted only for float in torch<=1.10.0
			img.float(),
			kernel,
			stride=self.stride,
			dilation=self.dilation
		)

	def __repr__(self):
		return str(self.kernel)

class CompositeConvolveKernel(torch.nn.Module):
	"""
		Kernel convolution module for multi-stage convolutions (e.g. vertical kernel following horizontal kernel). If given kernel is not instance of `ConvolveKernel`, constructs one with given kernel array.
	"""
	def __init__(self, kernels):
		super(CompositeConvolveKernel, self).__init__()
		if not isinstance(kernels, list):
			kernels = [kernels]
		layers = []
		for k in kernels:
			if isinstance(k, ConvolveKernel):
				layers.append(k)
			else:
				layers.append(ConvolveKernel(k))
		self.kernels = torch.nn.Sequential(*layers)

	def forward(self, img):
		return self.kernels(img)

class SobelKernel(CompositeConvolveKernel):
	"""
		Sobel's edge detector.
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

class RobertsKernel(CompositeConvolveKernel):
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

class BoxKernel(ConvolveKernel):
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

class GaussianKernel(ConvolveKernel):
	def __init__(self, size, std, **kwargs):
		k = torch.linspace(-1,1,size)
		gx, gy = torch.meshgrid(k, k)
		k = torch.exp(-0.5/std * (gx*gx+gy*gy))
		k /= 2. * math.pi * std
		print(k.min(), k.max())
		super(GaussianKernel, self).__init__(k, **kwargs)

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
	x = torch.tensor([[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]).float()
	print(x.int())
	k = BoxKernel(3)
	print(k(x))
	k = ErosionKernel(3)
	print(k(x).int())
	k = DilationKernel(3, binarize=False, padding_mode='replicate')
	print(k(x))