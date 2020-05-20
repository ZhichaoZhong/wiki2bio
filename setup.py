try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from setuptools import find_packages
setup(
    name='wikibio',
    version='0.0.1',
    url='https://github.com/ZhichaoZhong/wiki2bio',
    author='ZhichaoZhong',
    author_email='zzhong@wehkamp.nl',
    description='Package for wiki2bio, check the master branch for the orignal code.',
    install_requires = ['tensorflow==1.1.0'],
    packages=find_packages(exclude=['original_data', 'doc']),
    python_requires='>=3.7',
)