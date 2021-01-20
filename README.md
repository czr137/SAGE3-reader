# Python SAGE Data Reader

A python reader for SAGE III and SAGE III ISS binary data files into an xarray dataset.

Tested with versions:
* 5.0 
* 5.1
* 5.2

## Downloads 
The binary files are not supplied and must be downloaded by the user.  
[SAGEIII binary files](https://eosweb.larc.nasa.gov/project/sage3/sage3_table)  
[SAGEIII-ISS binary files](https://eosweb.larc.nasa.gov/project/sageiii-iss/sageiii-iss_table)

# Installation

To install the package run:
```
pip install sage3reader
```

# Dataset details

The user should be able to match the coordinates and variables in the dataset with those in the binary file format sheet
(Table C1.3 of the [Data Product User Guide](https://eosweb.larc.nasa.gov/sites/default/files/project/sage3/guide/Data_Product_User_Guide.pdf)) 
taking note of the following changes

Binary Format | Dataset
--- | ---
start_date (yyyymmdd) and start_time (hhmmss) integers | start_time (datetime64)
end_date (yyyymmdd) and end_time (hhmmss) integers | end_time (datetime64)
ground track date and time integers | ground track datetime64

# Examples

To load data for a single profile:

``` python
from sage3reader import l2binary_to_dataset

s3data = l2binary_to_dataset('/path/to/file')
```

To load all profiles in a directory tree:

``` python
from sage3reader import multi_path_l2binary_to_dataset

s3data = multi_path_l2binary_to_dataset('/path/to/files')
```

Other examples can be found in the `examples` folder.