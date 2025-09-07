from setuptools import setup, find_packages

VERSION = '0.0.1'

setup_info = dict(
    name='SciBuddy',
    version=VERSION,
    author='Juraj Fulir',
    author_email='jurajfulir@gmail.com',
    url='https://github.com/lirfu/SciBuddy',
    description='A Python toolkit of common utils for data scientists.',
    long_description='SciBuddy contains a bunch of tools used for configurable experiment tracking, advanced logging, elegant data loading and plotting and much more.',
    license='MIT',
    zip_safe=True,

	packages=find_packages(),

	install_requires=[
		'numpy>=1.22.3',
		'matplotlib>=3.2.2',
		'Pillow>=9.1.0',
		'torch>=1.11.0',
		'torchvision>=0.12.0',
		'tqdm>=4.64.0',
		'trimesh>=3.12.0',
		'opencv-python-headless>=4.5.5.64',
		'pyyaml>=6.0',
		'psutil>=5.9.0'
	]
)

setup(**setup_info)