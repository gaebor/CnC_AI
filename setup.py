# -*- coding: utf-8 -*-

from setuptools import setup
from subprocess import run, PIPE
from re import search

INSTALL_REQUIRES = ['numpy', 'tornado', 'termcolor', 'tqdm']
nvidia_smi_output = run(
    'nvidia-smi', text=True, shell=True, check=False, stdout=PIPE, stderr=PIPE
).stdout
version_match = search('CUDA Version: ((\d+)\.(\d+))', nvidia_smi_output)
if version_match:
    cuda_version_str = 'cu' + version_match.group(2) + version_match.group(3)
    dependency_links = ['https://download.pytorch.org/whl/' + cuda_version_str]
    INSTALL_REQUIRES.append(f'torch>=1.9.1+{cuda_version_str}')
else:
    dependency_links = []
    INSTALL_REQUIRES.append('torch')


setup(
    name='cnc_ai',
    version='0.1',
    description="A neural network to play Command & Conquer (Remastered)",
    long_description=open('README.md').read(),
    author="Gábor Borbély",
    author_email='borbely@math.bme.hu',
    url='https://github.com/gaebor/CnC_AI',
    license='MIT',
    install_requires=INSTALL_REQUIRES,
    packages=['cnc_ai', 'cnc_ai.TIBERIANDAWN'],
    include_package_data=True,
    dependency_links=dependency_links,
)
