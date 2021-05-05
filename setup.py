# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='cnc_ai',
    version='0.1',
    description="A neural network to play Command & Conquer Tiberian Dawn (Remastered)",
    long_description=open('README.md').read(),
    author="Gábor Borbély",
    author_email='borbely@math.bme.hu',
    url='https://github.com/gaebor/CnC_AI',
    license='MIT',
    install_requires=['numpy', 'torch', 'torchvision', 'Pillow'],
    packages=['cnc_ai'],
)