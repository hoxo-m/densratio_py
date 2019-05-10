# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='densratio',
    version='0.2.1',
    description='A Python Package for Density Ratio Estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hoxo-m/densratio_py',
    author='Koji Makiyama, Ameya Daigavane',
    author_email='hoxo.smile@gmail.com',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='density ratio, anomaly detection, change point detection, covariate shift',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy'],
    project_urls={
        'Bug Reports': 'https://github.com/hoxo-m/densratio_py/issues',
        'Say Thanks!': 'https://saythanks.io/to/hoxo-m',
        'Source': 'https://github.com/hoxo-m/densratio_py',
    },
    license="MIT + file LICENSE",
    test_suite='tests',
)
