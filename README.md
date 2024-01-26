
# FMACE

Foundation Model based on ACE Framework [arxiv:2310.02074](https://arxiv.org/abs/2310.02074) fine tuned with  MERRA2 data.

## Create conda environment

	> conda create --name fme --clone fmod
    > conda activate fme
    > pip install toolz dask dacite
    > git clone https://github.com/ai2cm/ace.git
    > cd FMACE
    > ln -s ./ace/fme/fme ./fme

	>> cd ../ace/fme
    >> pip install .

## Custom Modulus installation

    >> git clone https://github.com/NVIDIA/modulus.git
    >> cd modulus
    >> git checkout b8e27c5c4ebc409e53adaba9832138743ede2785
    >> git switch -c ace
    >> make install

## Run Tests

    >> cd FMACE
    >> pytest

## Run Inference

### Pretrained 

    >> cd FMACE
    >> python -m fmace.pipeline.inference config/explore-inference.yaml

### MERRA2 Fine-tuned 


