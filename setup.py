# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='densratio',
    version='0.0.1',
    description='A Python Package for Density Ratio Estimation',
    long_description=readme,
    author='Koji Makiyama',
    author_email='hoxo.smile@gmail.com',
    install_requires=['scipy'],
    url='https://github.com/hoxo-m/densratio_py',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
