from typing import List

import colorsys


def get_unique_color_from_hsv(i:int, N:int) -> List[int]:
	"""
		Generate a unique HSV color for index i out of N, and convert it to RGB.
	"""
	assert N < 256, 'Supporting only up to 255 unique colors!'
	color = colorsys.hsv_to_rgb(i/N, 1, 1)  # NOTE: Dividing by L because color for hue 0 and 1 are the same!
	return [int(color[0]*255.99), int(color[1]*255.99), int(color[2]*255.99)]

def hex_to_rgb(v:str) -> List[int]:
	"""
		Converts a hex RGB string to a list of integers (uint8).
	"""
	return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]

def rgb_to_hex(color:List[int]) -> str:
	"""
		Converts a list of integers (uint8) to a hex RGB string.
	"""
	assert color[0]>=0 and color[1]>=0 and color[2]>=0, 'Color values must be uint8!'
	assert color[0]<=255 and color[1]<=255 and color[2]<=255, 'Color values must be uint8!'
	return '%0.2X'%color[0] + '%0.2X'%color[1] + '%0.2X'%color[2]