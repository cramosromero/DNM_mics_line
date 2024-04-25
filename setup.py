# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:45:20 2023

@author: SES271
"""

from setuptools import setup, find_packages

setup(
    name='DNtools',
    version='1.0',
    description='Tools for Acoustic Characterisation of sUAS from on-field measurments. 9 ground-plate microphone array',
    author='Carlos Ramos-Romero',
    author_email='c.a.ramosromero@salford.ac.uk',
    packages=["dntools"],
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: DroneNoise EPSRC users',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy', 'mpmath', 'scipy', 'matplotlib'
    ],
)