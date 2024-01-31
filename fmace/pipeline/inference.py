import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence
from fme.core.data_loading.requirements import DataRequirements
import dacite
import torch
import yaml, hydra
from fmace.pipeline.config import InferenceConfig
import fme
from fme.fcn_training.registry import ModuleSelector
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.data_loading.get_loader import get_data_loader
from fmod.base.util.config import cfg, cfg2meta, configure
from fme.core.wandb import WandB
from fme.fcn_training.inference.loop import run_dataset_inference, run_inference
from fme.fcn_training.utils import gcs_utils, logging_utils
from fmace.pipeline.config import get_stepper_config


def main( configuration: str ):

    hydra.initialize(version_base=None, config_path="../config")
    configure( configuration )

 #   if not os.path.isdir(config.experiment_dir):
 #       os.makedirs(config.experiment_dir)
 #   config.configure_logging(log_filename="inference_out.log")
 #   config.configure_wandb()
 #   gcs_utils.authenticate()

  #  torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging_utils.log_beaker_url()
    logging_utils.log_slurm_info()

    builder = ModuleSelector(type=cfg().pipline.module, config=cfg().pipline),
    stepper_config = get_stepper_config(builder)
    logging.info("Loading inference data")
    data_requirements: DataRequirements = stepper_config.get_data_requirements( n_forward_steps=cfg().model.n_forward_steps )
    config = InferenceConfig.get_instance()

    def _get_data_loader(window_time_slice: Optional[slice] = None):
        """
        Helper function to keep the data loader configuration static,
        ensuring we get the same batches each time a data loader is
        retrieved, other than the choice of window_time_slice.
        """
        return get_data_loader(
            config.validation_data,
            requirements=data_requirements,
            train=False,
            window_time_slice=window_time_slice,
        )

    # use window_time_slice to avoid loading a large number of timesteps
    validation = _get_data_loader(window_time_slice=slice(0, 1))

    stepper = stepper_config.load_stepper(
        validation.area_weights.to(fme.get_device()),
        sigma_coordinates=validation.sigma_coordinates.to(fme.get_device()),
    )

    aggregator = InferenceAggregator(
        validation.area_weights.to(fme.get_device()),
        sigma_coordinates=validation.sigma_coordinates,
        record_step_20=config.n_forward_steps >= 20,
        log_video=config.log_video,
        enable_extended_videos=config.log_extended_video,
        log_zonal_mean_images=config.log_zonal_mean_images,
        n_timesteps=config.n_forward_steps + 1,
        metadata=validation.metadata,
    )
    writer = config.get_data_writer(validation)

    def data_loader_factory(window_time_slice: Optional[slice] = None):
        return _get_data_loader(window_time_slice=window_time_slice)

    logging.info("Starting inference")
    if config.prediction_data is not None:
        # define data loader factory for prediction data
        def prediction_data_loader_factory(window_time_slice: Optional[slice] = None):
            return get_data_loader(
                config.prediction_data,
                requirements=data_requirements,
                train=False,
                window_time_slice=window_time_slice,
            )

        run_dataset_inference(
            aggregator=aggregator,
            normalizer=stepper.normalizer,
            prediction_data_loader_factory=prediction_data_loader_factory,
            target_data_loader_factory=data_loader_factory,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
            writer=writer,
        )
    else:
        run_inference(
            aggregator=aggregator,
            writer=writer,
            stepper=stepper,
            data_loader_factory=data_loader_factory,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
        )

    logging.info("Starting logging of metrics to wandb")
    step_logs = aggregator.get_inference_logs(label="inference")
    wandb = WandB.get_instance()
    for i, log in enumerate(step_logs):
        wandb.log(log, step=i)
        # wandb.log cannot be called more than "a few times per second"
        time.sleep(0.3)
    writer.flush()

    logging.info("Writing reduced metrics to disk in netcdf format.")
    for name, ds in aggregator.get_datasets(("time_mean", "zonal_mean")).items():
        coords = {k: v for k, v in validation.coords.items() if k in ds.dims}
        ds = ds.assign_coords(coords)
        ds.to_netcdf(Path(config.experiment_dir) / f"{name}_diagnostics.nc")
    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=str)
    args = parser.parse_args()

    main( configuration=args.configuration )
