from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sage3reader',
    version='0.2.5',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='https://github.com/czr137/SAGE3-reader',
    project_urls={
        'Group Homepage': 'https://research-groups.usask.ca/osiris/',
    },
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
