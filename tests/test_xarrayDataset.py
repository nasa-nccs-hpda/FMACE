import os, dataclasses, logging, time
from pathlib import Path
from typing import Optional, Sequence
import dacite
import yaml
from fme.fcn_training.inference.inference import InferenceConfig
from fme.core.data_loading._xarray import XarrayDataset

from fme.core.data_loading.requirements import DataRequirements
from fme.core.stepper import SingleModuleStepperConfig

#from fme.core.data_loading.params import DataLoaderParams
#from fme.fcn_training.utils import gcs_utils, logging_utils

yaml_config = Path(__file__).parent.parent.joinpath("ace_config/explore-inference.yaml").resolve()
with open(yaml_config, "r") as f:
	data = yaml.safe_load(f)
config: InferenceConfig = dacite.from_dict( data_class=InferenceConfig, data=data, config=dacite.Config(strict=True) )

window_time_slice: Optional[slice] = None # slice()
if not os.path.isdir(config.experiment_dir): os.makedirs(config.experiment_dir)
with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
	yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(config)

# config.configure_logging(log_filename="inference_out.log")
# config.configure_wandb()
# gcs_utils.authenticate()
#
# logging_utils.log_versions()
# logging_utils.log_beaker_url()
# logging_utils.log_slurm_info()
print("Config loaded")

stepper_config: SingleModuleStepperConfig = config.load_stepper_config()
data_requirements: DataRequirements = stepper_config.get_data_requirements( n_forward_steps=config.n_forward_steps )

print(f"\n * normalization: {stepper_config.normalization}")
print(f"\n * stepper_config: {stepper_config}" )
print(f"\n * data_requirements: {data_requirements}" )

dataset = XarrayDataset( config.validation_data, requirements=data_requirements, window_time_slice=window_time_slice )
print( dataset )

