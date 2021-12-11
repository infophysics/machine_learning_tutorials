# Machine Learning Tutorials

A collection of machine learning tutorials in python.

### Installation
----------------
In order to utilize the jupyter notebooks in this package, we recommend installing the IDE VS code (visual studio code), which can be integrated with anaconda environments and also contains it's own jupyter interface rather than having to use the browser.  You can also connect to remote environments, which makes integrating jupyter notebooks with remote sessions seamless.  You can download **VS code** [here](https://code.visualstudio.com/download).

#### Packages
-------------
# <a href="https://numpy.org/"><img alt="NumPy" src="/branding/logo/primary/numpylogo.svg" height="60"></a>
![JAX](https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png)

For the tutorials you will need to install the following basic packages:
1. Anaconda ([individual edition](https://www.anaconda.com/products/individual)).
2. NumPy ([latest version](https://numpy.org/)) - `conda install numpy`.
3. SciPy ([latest version](https://scipy.org/)) - `conda install scipy`.
4. Matplotlib ([latest version](https://matplotlib.org/)) - `conda install matplotlib`.
5. Pandas ([latest version](https://pandas.pydata.org/)) - `conda install pandas`.
6. Scikit-learn (with parallel processing) ([latest version](https://scikit-learn.org/stable/index.html)) - `conda install scikit-learn scikit-learn-intelex`.
7. Jupyter ([latest version](https://jupyter.org/)) - `conda install jupyter`.
8. PyTorch ([latest version](https://pytorch.org/)) - `conda install -c pytorch`.
9. JAX ([latest version](https://github.com/google/jax)) - `pip install jax && pip install jaxlib`.

For information on how to use **anaconda from within jupyter** see [this page](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/).

#### Anaconda
-------------
To install anaconda on Linux, you can download the shell installer [here](https://www.anaconda.com/products/individual), or issue the following from a shell
```bash
wget -O - https://www.anaconda.com/distribution/ 2>/dev/null | sed -ne 's@.*\(https:\/\/repo\.anaconda\.com\/archive\/Anaconda3-.*-Linux-x86_64\.sh\)\">64-Bit (x86) Installer.*@\1@p' | xargs wget
```
You'll then need to change permissions on the file you just downloaded,
```bash
chmod +x Anaconda*.sh
```
and then run the installer,
```bash
./Anaconda*.sh
```
##### Creating a custom anaconda environment
The installer will ask you to read through and accept a license, and will then prompt you for an installation directory (by default it will install in ~/anaconda3).  Once it's installed, you can create custom anaconda environments by issuing the command
```bash
conda create -n "name" python="python_version"
```
where *name* is the name you want to give to your environment and *python_version* is the python version you want to install, typically 3.7 or 3.8 (3.9 is currently the bleeding edge which may not be compatible with some packages).  For the tutorials it could be helpful to make a custom environment with python version 3.8:
```bash
conda create -n ml_tutorials python=3.8
```
##### Loading your custom environment
To load the environment you just created, issue the command
```bash
conda activate ml_tutorials
```
You can see which packages are active by issuing:
```bash
conda list -n ml_tutorials
```
For more on managing anaconda environments, see the docs [here](https://docs.conda.io/projects/conda/en/latest/index.html) and the [tutorials](https://docs.anaconda.com/anaconda/navigator/tutorials/index.html).

#### Python packages
--------------------
Once you are in an anaconda environment, you can install packages such as NumPy using the command
```bash
conda install numpy
```
which will collect a set of prerequisite packages that also need to be installed alongside numpy.  The installer will prompt you to agree to any packages before it installs them.  Not all python packages can be found by the conda installer, and so you may have to resort to using [pip](https://pypi.org/project/pip/).  Several packages can be installed at once
```bash
conda install -c numpy scipy matplotlib pandas scikit-learn scikit-learn-intelex jupyter pytorch

```
#### Jupyter
------------
Once jupyter is installed, you'll want to create a kernel which is associated to your anaconda environment that can be used within jupyter notebooks.  To do this, simply issue the command (assuming the preferred anaconda environment is active)
```bash
python -m ipykernel install --user --name "anaconda_environment" --display-name "Python (anaconda_environment)"
```
where *anaconda_environment* is the name of the environment you want to create a kernel for and "Python (anaconda_environment)" is the given display name.  Assuming you used "ml_tutorials" for your environment, the command would be
```bash
python -m ipykernel install --user --name ml_tutorials --display-name "Python (ML tutorials)"
```
Once inside of a jupyter notebook, you can check to see which kernel is currently being used by issuing
```python
import os
print(os.environ['CONDA_DEFAULT_ENV'])
```
which will print the name of the anaconda environment.

#### VS Code
------------
VS Code is extremely useful since it can easily integrate with jupyter and anaconda environments, as well as any remote ssh environments where you want to do your development.  To configure ssh with VS Code, see [this link](https://code.visualstudio.com/docs/remote/ssh).  

To configure jupyter, first download the standard python extension for VS Code by typing ```Ctrl+Shift+P``` which will bring up a search window where you can type commands or search for things within the IDE.  Type ```Extensions: Install Extensions``` and select the option once it becomes available.  This will bring up a side bar that includes a search window.  Type ```ms-python.python``` in the search window and install the result.

Once the python extension is installed, you should be able to select the interpreter by doing ```Ctrl+Shift+P``` and typing ```Python: Select Interpreter```.  This will bring up a list (hopefully) of python kernels including any anaconda environments.  Selecting the environment you created will allow you to run python scripts from VS Code which use that environment.

For more on using jupyter with VS Code see [this link](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
