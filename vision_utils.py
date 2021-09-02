import torch

def apply_kernel(img, kernel):
	pad = int((kernel.shape[-1] - 1) / 2)
	if len(kernel.shape) == 2:
		kernel = kernel.unsqueeze(0).unsqueeze(0).to(img.device)
	elif len(kernel.shape) == 3:
		kernel = kernel.unsqueeze(0).to(img.device)
	else:
		raise RuntimeError('Valid kernel must have 2 or 3 dimensions.')
	if img.shape[1] != kernel.shape[1]:
		kernel = kernel.repeat(1,img.shape[1],1,1)
	return torch.nn.functional.conv2d(img, weight=kernel, padding=pad)

sobel_kernel_x = torch.tensor([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]
]).float()
sobel_kernel_y = torch.tensor([
	[1, 2, 1],
	[0, 0, 0],
	[-1, -2, -1]
]).float()

def apply_sobel(img):
	img = apply_kernel(img, sobel_kernel_x)
	return apply_kernel(img, sobel_kernel_y)

roberts_kernel_x = torch.tensor([
	[1, 0],
	[0, -1]
]).float()
roberts_kernel_y = torch.tensor([
	[0, 1],
	[-1, 0]
]).float()

def apply_roberts(img):
	img = apply_kernel(img, roberts_kernel_x)
	return apply_kernel(img, roberts_kernel_y)

def make_box_kernel(size):
	return torch.ones(size,size, dtype=torch.float) / size**2

def apply_blurred_edge(img, img_b, highlight_inside=True, highlight_outside=True):
	if highlight_inside and highlight_outside:
		return img + img_b - 2. * img * img_b
	elif highlight_inside:
		return img * (1-img_b)
	elif highlight_outside:
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