import torch

class ConvolveKernel(torch.Module):
	def __init__(self, kernel):
		super(ConvolveKernel, self).__init__()
		if len(self.kernel.shape) == 2:
			kernel = kernel.unsqueeze(0).unsqueeze(0)
		elif len(kernel.shape) == 3:
			kernel = kernel.unsqueeze(0)
		else:
			raise RuntimeError('Valid kernel must have 2 or 3 dimensions.')
		self.kernel = kernel
		self.padding = int((self.kernel.shape[-1] - 1) / 2)		

	def forward(self, img):
		kernel = self.kernel
		if img.shape[1] != kernel.shape[1]:  # Apply same kernel across multiple channels.
			kernel = kernel.repeat(1,img.shape[1],1,1)
		return torch.nn.functional.conv2d(img, weight=kernel, padding=self.padding)

class CompositeConvolveKernel(torch.Module):
	def __init__(self, kernels):
		super(CompositeConvolveKernel, self).__init__()
		if not isinstance(kernels, list):
			kernels = [kernels]
		layers = []
		for k in kernels:
			layers.append(ConvolveKernel(k))
		self.kernels = torch.nn.Sequential(*layers)

	def forward(self, img):
		return self.kernels(img)

class SobelKernel(CompositeConvolveKernel):
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
	def __init__(self, size):
		super(BoxKernel, self).__init__(
			torch.ones(size,size, dtype=torch.float) / size**2
		)

class BlurEdgeDetector(torch.Module):
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