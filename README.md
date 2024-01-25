
# FMACE

Foundation Model based on ACE Framework [arxiv:2310.02074](https://arxiv.org/abs/2310.02074) fine tuned with  MERRA2 data.

## Create conda environment

	> conda create --name fme --clone fmod
    > conda activate fme
    > pip install toolz dask
    > git clone https://github.com/ai2cm/ace.git
    > cd FMACE
    > ln -s ./ace/fme/fme ./fme

	>> cd ../ace/fme
    >> pip install .


