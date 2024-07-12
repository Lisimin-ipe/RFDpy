from setuptools import setup, find_packages
import sklearn

setup(
    name='RFDpy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
    ],
    author='Simin Li',
    author_email='smli@ipe.ac.cn',
    description='A package for machine learning model evaluation and data processing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lisimin-ipe/RFDpy', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
