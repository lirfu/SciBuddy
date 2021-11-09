import math
import torch


def chw_to_hwc(img):
	return img.moveaxis(0,2)

def hwc_to_chw(img):
	return img.moveaxis(2,0)

class ConvolveKernel(torch.nn.Module):
	"""
		Simple kernel convolution module.
	"""
	def __init__(self, kernel, padding=None, padding_mode='constant', padding_value=0, stride=1, dilation=1):
		super(ConvolveKernel, self).__init__()
		self.kernel = kernel
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

class IdentityKernel(ConvolveKernel):
	def __init__(self, **kwargs):
		super(IdentityKernel, self).__init__(torch.tensor([[1.]]), **kwargs)

class SobelKernel(CompositeConvolveKernel):
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
	
	def forward(self, img):
		return super().forward(img) / 2

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
		# print(k.min(), k.max())  # TODO
		super(GaussianKernel, self).__init__(k, **kwargs)

class VerticalEdgeKernel(ConvolveKernel):
	def __init__(self, **kwargs):
		super(VerticalEdgeKernel, self).__init__(torch.tensor([
			[-1,0,1],
			[-1,0,1],
			[-1,0,1]
		], dtype=torch.float32) / 3, **kwargs)

class HorizontalEdgeKernel(ConvolveKernel):
	def __init__(self, **kwargs):
		super(HorizontalEdgeKernel, self).__init__(torch.tensor([
			[-1,-1,-1],
			[0,0,0],
			[1,1,1]
		], dtype=torch.float32) / 3, **kwargs)

class RotateKernel(ConvolveKernel):
	def __init__(self, kernel, phi, **kwargs):
		kernel = kernel.kernel.squeeze(0).squeeze(0)
		s, c = -math.sin(phi), math.cos(phi)
		h, w = (kernel.shape[-2]-1)/2, (kernel.shape[-1]-1)/2
		rotator = torch.tensor([
			[c, -s, -c*w + s*h + w],
			[s,  c, -s*w - c*h + h],
			[0, 0, 1]
		])
		h, w = kernel.shape[-2], kernel.shape[-1]
		indices = torch.stack([torch.arange(w).reshape(-1,1).repeat(1,h), torch.arange(h).reshape(1,-1).repeat(w,1), torch.ones((w,h))], dim=2)
		end_indices = torch.stack([rotator.matmul(i) for r in indices for i in r])
		m = end_indices.floor().long().clip(torch.tensor([0,0,0]),torch.tensor([h-1,w-1,1]))
		M = end_indices.round().long().clip(torch.tensor([0,0,0]),torch.tensor([h-1,w-1,1]))
		kk = torch.zeros(h*w)
		kk += kernel[m[:,0], m[:,1]]
		kk += kernel[m[:,0], M[:,1]]
		kk += kernel[M[:,0], m[:,1]]
		kk += kernel[M[:,0], M[:,1]]
		kk = kk.reshape(h,w) / 4
		self.m, self.M = torch.relu(-kk).sum(), torch.relu(kk).sum()
		super(RotateKernel, self).__init__(kk)

	def forward(self, img):
		return (super().forward(img) - self.m) / (self.M - self.m)

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
	# Display kernel functions.
	from tools import show_images
	kernels = {
		'Identity': IdentityKernel(),
		# 'Sobel': SobelKernel(),
		# 'Roberts': RobertsKernel(),
		# 'Box': BoxKernel(3),
		# 'Erosion': ErosionKernel(3),
		# 'Dilation': DilationKernel(3),
		# 'Gaussian': GaussianKernel(3, 1),

		'Vertical edges': VerticalEdgeKernel(),
		'Rotation 15': RotateKernel(VerticalEdgeKernel(), -math.pi / 12),
		'Rotation 30': RotateKernel(VerticalEdgeKernel(), -math.pi / 6),
		'Rotation 45': RotateKernel(VerticalEdgeKernel(), -math.pi / 4),
		'Rotation 60': RotateKernel(VerticalEdgeKernel(), -math.pi / 3),
		'Rotation 75': RotateKernel(VerticalEdgeKernel(), -math.pi * 5 / 12),
		'Rotation 90': RotateKernel(VerticalEdgeKernel(), -math.pi / 2),
		'Horizontal edges': HorizontalEdgeKernel()
	}
	img = torch.zeros((32,32), dtype=torch.float32)
	img[0,:] = 1
	img[:6,0] = 1
	img[:6,-1] = 1
	img[10:21, 5:11] = 1
	img[8, 20] = 1
	img[20, 15:26] = 1
	img[15:26, 20] = 1
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