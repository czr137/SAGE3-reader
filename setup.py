from setuptools import setup

setup(
    name='sage3reader',
    version='0.1.0',
    packages=['sage3reader'],
    url='http://arg.usask.ca',
    license='MIT',
    author='Chris Roth',
    author_email='chris.roth@usask.ca',
    description='A python reader for SAGE III and SAGE III ISS L2 solar binary files',
    keywords='sage sageiii reader l2 binary solar',
    python_requires='>=3.6',
    install_requires=['numpy', 'xarray>=0.10']
)
