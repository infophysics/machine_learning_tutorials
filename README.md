# Machine Learning Tutorials

A collection of machine learning tutorials in python.

### Installation

In order to utilize the jupyter notebooks in this package, we recommend installing the IDE VS code (visual studio code), which can be integrated with anaconda environments and also contains it's own jupyter interface rather than having to use the browser.  You can also connect to remote environments, which makes integrating jupyter notebooks with remote sessions seamless.  You can download VS code [here](https://code.visualstudio.com/download).

For the tutorials you will need to install the following basic packages:
1. Anaconda ([individual edition](https://www.anaconda.com/products/individual)).
2. NumPy ([latest version](https://numpy.org/)) - `conda install numpy`.
3. SciPy ([latest version](https://scipy.org/)) - `conda install scipy`.
4. PyTorch ([latest version](https://pytorch.org/)) - `conda install pytorch`.

#### Anaconda
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
conda create -n <name> python=<python_version>
```
where <name> is the name you want to give to your environment and <python_version> is the python version you want to install, typically 3.7 or 3.8 (3.9 is currently the bleeding edge which may not be compatible with some packages).  For the tutorials it could be helpful to make a custom environment with python version 3.8,
```bash
conda create -n ml_tutorials python=3.8
```
##### Loading your custom environment

#### Python packages
Once you have anaconda installed, 


