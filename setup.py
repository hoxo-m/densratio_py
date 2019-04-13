# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='densratio',
    version='0.2',
    description='A Python Package for Density Ratio Estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Koji Makiyama',
    author_email='hoxo.smile@gmail.com',
    install_requires=['numpy'],
    url='https://github.com/hoxo-m/densratio_py',
    license="MIT + file LICENSE",
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
