# jupyter-ilab
Jupyterlab Tools and Widgets developed for the NASA-NCCS Innovation Lab

### Installation
These instructions assume that jupyterlab has already been installed.  
*jupyterlab_env* is a conda environment that is used in jupyterlab.

* **Add jupyterlab extensions** (if not already present):
```
>> jupyter labextension install @jupyter-widgets/jupyterlab-manager
>> jupyter labextension install jupyter-matplotlib
```   

* **Add conda dependencies**

```
>> conda activate jupyterlab_env
(jupyterlab_env)>> conda install -c conda-forge xarray dask matplotlib numpy pandas scikit-learn netCDF4 ipympl nodejs
```    

* **Install jupyter-ilab**

```
    (jupyterlab_env)>> git clone https://github.com/nasa-nccs-cds/jupyter-ilab.git jupyter_ilab
    (jupyterlab_env)>> cd jupyter_ilab
    (jupyterlab_env)>> python setup.py install
```

* **Startup jupyter-ilab** 

```
    (jupyterlab_env)>> cd notebooks
    (jupyterlab_env)>> jupyter lab
```
* **Run the LabDemo.ipynb demo notebook**