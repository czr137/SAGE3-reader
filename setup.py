from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sage3reader',
    version='0.2.4',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='http://arg.usask.ca',
    license='MIT',
    author='Chris Roth',
    author_email='chris.roth@usask.ca',
    description='A python reader for SAGE III and SAGE III ISS L2 solar binary files',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='sage sageiii reader l2 binary solar',
    python_requires='>=3.6',
    install_requires=['numpy>=1.15', 'xarray>=0.11']
)
