import cv2
import torch

from .data import torch_float_to_torch_uint8

class ComputerCamera:
	def __init__(self, idx=0):
		self.cam = cv2.VideoCapture(idx)

		if not self.cam.isOpened():
			raise RuntimeError('Could not open camera!')

	def __del__(self):
		self.cam.release()

	def set_res(self, width=640, height=480):
		self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
		self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

	def get_frame(self):
		rval, frame = self.cam.read()
		return frame if rval else None


def cvframe_to_torch_float(frame):
	return (torch.tensor(frame, dtype=torch.float) * (1./255)).permute(2,0,1)

def torch_float_to_cvframe(img):
	img = torch_float_to_torch_uint8(img).permute(1,2,0)
	return cv2.UMat(img.detach().cpu().numpy())


if __name__ == '__main__':
	# Play around with object detection on streaming camera.
	cv2.namedWindow("preview")

	import time
	import torch
	import torchvision
	
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model.eval()
	model.to(device)
	COCO_INSTANCE_CATEGORY_NAMES = [
		'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
		'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
		'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
		'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
		'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
		'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
		'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
		'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
		'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
	]

	cam = ComputerCamera()

	def get_img(cam, model):
		frame = cam.get_frame()
		if frame is None:
			print('Could not get frame from camera!')
			return None
		
		# Object detection.
		img = cvframe_to_torch_float(frame).unsqueeze(0).to(device)
		pred = model(img)
		indices = pred[0]['scores'] > 0.8
		boxes = pred[0]['boxes'][indices].cpu()
		labels = pred[0]['labels'][indices].cpu()
		for b,l in zip(boxes,labels):
			cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
			cv2.putText(frame, COCO_INSTANCE_CATEGORY_NAMES[l.item()], (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
		return frame

	t = time.time()
	frame = get_img(cam, model)
	while frame is not None:
		cv2.imshow("preview", frame)

		key = cv2.waitKey(1)
		if key == 27: # exit on ESC
			break

		dt = time.time()-t
		t = time.time()
		print('{: 8.2f} ms ({: 5.2f} FPS)'.format(dt*1000., 1./dt))
		frame = get_img(cam, model)

	cv2.destroyWindow("preview")
