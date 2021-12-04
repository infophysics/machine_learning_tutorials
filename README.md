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
```shell
wget -O - https://www.anaconda.com/distribution/ 2>/dev/null | sed -ne 's@.*\(https:\/\/repo\.anaconda\.com\/archive\/Anaconda3-.*-Linux-x86_64\.sh\)\">64-Bit (x86) Installer.*@\1@p' | xargs wget
```


