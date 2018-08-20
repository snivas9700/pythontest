### Price (discount) optimization for Technology Support Services (TSS) offerings bundled with IBM Power and Storage Systems

Documentation of the codebase is available on [to be constructed Github Pages 
site]
(to be added).

This codebase is built with Python 3.6 in modules of functions, some of which are nested.

It can be run by executing portions of the script `main`, either in an IDE like Pycharm or in a Jupyter notebook (either locally or on [IBM DSX Local](https://9.220.2.27/); [how to access DSX Local](https://github.ibm.com/cognitive-data-platform/cognitive-data-platform/tree/master/documentation/DSX_Local)).

I strongly recommend cloning the repository into Pycharm and then pushing 
changes to your own dev branch.  Detailed step-by-step instructions on how 
to set up Pycharm to clone and version-control in this way are provided in 
a Word document on the repo.

Alternatively, in a notebook, import the necessary modules using the script
 in `main.py`.

### Overview of how the main script runs the software
1. Load libraries needed to perform a range of tasks, including Pandas, and NumPy, as well as modules `cleansing` and `descriptive`.

2. Load data from a local folder. Note that you must modify the names of paths, including `path_root`, `path_in`, and `path_out`.

3. Prepare multiple data sets using functions in `cleansing` and merge into one data set.

4. Perform descriptive analyses, either in-line or through functions provided in `descriptive`.  A library of custom-built visualization functions is provided in `myplotlibrary4_3`

5. To date, an unsupervised learning (clustering) function is available in `descriptive`; I recommend adding linear regression functions to this same module. 