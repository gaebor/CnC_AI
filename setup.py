# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='cnc_ai',
    version='0.1',
    description="A neural network to play Command & Conquer (Remastered)",
    long_description=open('README.md').read(),
    author="Gábor Borbély",
    author_email='borbely@math.bme.hu',
    url='https://github.com/gaebor/CnC_AI',
    license='MIT',
    install_requires=['numpy', 'torch', 'torchvision', 'Pillow', 'tornado'],
    packages=['cnc_ai', 'cnc_ai.TIBERIANDAWN'],
    include_package_data=True,
)
