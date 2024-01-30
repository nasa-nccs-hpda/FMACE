from fme.fcn_training.registry import ModuleSelector
from fme.core.stepper import SingleModuleStepperConfig
from typing import Dict, List, Union, Optional, Sequence
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.loss import ConservationLoss, ConservationLossConfig, LossConfig
from fme.core.stepper import CorrectorConfig
import dataclasses
from fme.core.prescriber import Prescriber
import dataclasses
import logging
from typing import Optional, Sequence
import torch
import hydra
from fme.core import SingleModuleStepper
from fme.core.data_loading.params import DataLoaderParams
from fme.core.data_loading.typing import GriddedData, SigmaCoordinates
from fme.core.dicts import to_flat_dict
from fmod.base.util.config import cfg, cfg2meta, configure
from fme.fcn_training.inference.data_writer import DataWriter
from fme.fcn_training.train_config import LoggingConfig


def get_stepper_config() -> SingleModuleStepperConfig:
    stepper_config: SingleModuleStepperConfig = SingleModuleStepperConfig(
        builder = ModuleSelector( type="SphericalFourierNeuralOperatorNet", config=cfg().module ),
        in_names = cfg().model.in_names,
        out_names = cfg().model.out_names,
        normalization = cfg2meta('norm', NormalizationConfig() ),
        optimization = None,
        corrector = CorrectorConfig(conserve_dry_air=False, zero_global_mean_moisture_advection=False),
        prescriber = cfg2meta( 'prescriber', PrescriberConfig() ),
        loss = LossConfig(type='LpLoss', kwargs={}, global_mean_type=None, global_mean_kwargs={}, global_mean_weight=1.0) )
    return stepper_config


@dataclasses.dataclass
class PrescriberConfig:
    """
    Configuration for overwriting predictions of 'prescribed_name' by target values.

    If interpolate is False, the data is overwritten in the region where
    'mask_name' == 'mask_value' after values are cast to integer. If interpolate
    is True, the data is interpolated between the predicted value at 0 and the
    target value at 1 based on the mask variable, and it is assumed the mask variable
    lies in the range from 0 to 1.

    Attributes:
        prescribed_name: Name of the variable to be overwritten.
        mask_name: Name of the mask variable.
        mask_value: Value of the mask variable in the region to be overwritten.
        interpolate: Whether to interpolate linearly between the generated and target
            values in the masked region, where 0 means keep the generated values and
            1 means replace completely with the target values. Requires mask_value
            be set to 1.
    """

    prescribed_name: str = None
    mask_name: str = None
    mask_value: int = 1
    interpolate: bool = False

    def __post_init__(self):
        if self.interpolate and self.mask_value != 1:
            raise ValueError(
                "Interpolation requires mask_value to be 1, but it is set to "
                f"{self.mask_value}."
            )

    def build(self, in_names: List[str], out_names: List[str]):
        if not (self.prescribed_name in in_names and self.prescribed_name in out_names):
            raise ValueError(
                "Variables which are being prescribed in masked regions must be in"
                f" in_names and out_names, but {self.prescribed_name} is not."
            )
        return Prescriber(
            prescribed_name=self.prescribed_name,
            mask_name=self.mask_name,
            mask_value=self.mask_value,
            interpolate=self.interpolate,
        )

@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Attributes:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for. Must be divisble
            by forward_steps_in_memory.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: configuration for logging.
        validation_data: Configuration for validation data.
        prediction_data: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_extended_video_netcdfs: Whether to log videos of the predictions with
            statistical metrics as netcdf files.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
            time dimension.
        save_prediction_files: Whether to save the predictions as a netcdf file.
        save_raw_prediction_names: Names of variables to save in the predictions
             netcdf file. Ignored if save_prediction_files is False.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    validation_data: DataLoaderParams
    prediction_data: Optional[DataLoaderParams] = None
    log_video: bool = True
    log_extended_video: bool = False
    log_extended_video_netcdfs: bool = False
    log_zonal_mean_images: bool = True
    save_prediction_files: bool = True
    save_raw_prediction_names: Optional[Sequence[str]] = None
    forward_steps_in_memory: int = 1

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self):
        self.logging.configure_wandb(
            config=to_flat_dict(dataclasses.asdict(self)), resume=False
        )

    def configure_gcs(self):
        self.logging.configure_gcs()

    def load_stepper(
        self, area: torch.Tensor, sigma_coordinates: SigmaCoordinates
    ) -> SingleModuleStepper:
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
            sigma_coordinates: The sigma coordinates of the model.
        """
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return _load_stepper(
            self.checkpoint_path,
            area=area,
            sigma_coordinates=sigma_coordinates,
        )


    def get_data_writer(self, validation_data: GriddedData) -> DataWriter:
        n_samples = get_n_samples(validation_data.loader)
        return DataWriter(
            path=self.experiment_dir,
            n_samples=n_samples,
            n_timesteps=self.n_forward_steps + 1,
            metadata=validation_data.metadata,
            coords=validation_data.coords,
            enable_prediction_netcdfs=self.save_prediction_files,
            save_names=self.save_raw_prediction_names,
            enable_video_netcdfs=self.log_extended_video_netcdfs,
        )


def get_n_samples(data_loader):
    n_samples = 0
    for batch in data_loader:
        n_samples += next(iter(batch.data.values())).shape[0]
    return n_samples




