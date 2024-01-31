from fme.fcn_training.registry import ModuleSelector
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.loss import ConservationLoss, ConservationLossConfig, LossConfig
from fme.core.stepper import CorrectorConfig
import dataclasses, logging, torch, hydra
from fme.core.prescriber import Prescriber
from fme.core.data_loading.typing import GriddedData, SigmaCoordinates
from fme.core.dicts import to_flat_dict
from fmod.base.util.config import cfg, cfg2meta, configure
from fme.fcn_training.inference.data_writer import DataWriter
from fme.fcn_training.train_config import LoggingConfig
from typing import Literal, Optional, Sequence, List
from fme.core.distributed import Distributed


@dataclasses.dataclass
class Slice:
    """
    Configuration of a python `slice` built-in.

    Required because `slice` cannot be initialized directly by dacite.
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclasses.dataclass
class DataLoaderParams:
    """
    Attributes:
        data_path: Path to the data.
        data_type: Type of data to load.
        batch_size: Batch size.
        num_data_workers: Number of parallel data workers.
        n_repeats: Number of times to repeat the dataset (in time).
        n_samples: Number of samples to load, starting at the beginning of the data.
            If None, load all samples.
        window_starts: Slice indicating the set of indices to consider for initial
            conditions of windows of data. Values following the initial condition will
            still come from the full dataset. By default load all initial conditions.
        engine: Backend for xarray.open_dataset. Currently supported options
            are "netcdf4" (the default) and "h5netcdf". Only valid when using
            XarrayDataset.
    """

    data_path: str = ""
    data_type: Literal["xarray", "ensemble_xarray"] = "ensemble_xarray"
    batch_size: int = 1
    num_data_workers: int = 1
    n_repeats: int = 1
    n_samples: Optional[int] = None
    window_starts: Slice = dataclasses.field(default_factory=Slice)
    engine: Optional[Literal["netcdf4", "h5netcdf"]] = None

    def __post_init__(self):
        if self.n_samples is not None and self.batch_size > self.n_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be less than or equal to "
                f"n_samples ({self.n_samples}) or no batches would be produced"
            )
        if self.data_type not in ["xarray", "ensemble_xarray"]:
            if self.engine is not None:
                raise ValueError(
                    f"Got engine={self.engine}, but "
                    f"engine is unused for data_type {self.data_type}. "
                    "Did you mean to use data_type "
                    '"xarray" or "ensemble_xarray"?'
                )
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )

        if self.n_repeats != 1 and self.data_type == "ensemble_xarray":
            raise ValueError("n_repeats must be 1 when using ensemble_xarray")

def get_stepper_config(builder: ModuleSelector) -> SingleModuleStepperConfig:
    config = cfg().model
    print( f"get_stepper_config: norm config= {config.norm}")
    stepper_config: SingleModuleStepperConfig = SingleModuleStepperConfig(
        builder = builder,
        in_names = config.in_names,
        out_names = config.out_names,
        normalization = cfg2meta( config.norm, NormalizationConfig() ),
        optimization = None,
        corrector = CorrectorConfig(conserve_dry_air=False, zero_global_mean_moisture_advection=False),
        prescriber = cfg2meta( config.prescriber, PrescriberConfig() ),
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

    experiment_dir: str = None
    n_forward_steps: int = 1
    checkpoint_path: str = None
    log_video: bool = True
    log_extended_video: bool = False
    log_extended_video_netcdfs: bool = False
    log_zonal_mean_images: bool = True
    save_prediction_files: bool = True
    save_raw_prediction_names: Optional[Sequence[str]] = None
    forward_steps_in_memory: int = 1
    logging: LoggingConfig = None
    validation_data: DataLoaderParams = None
    prediction_data: Optional[DataLoaderParams] = None

    @classmethod
    def get_instance(cls):
        icfg = cfg2meta("inference", InferenceConfig() )
        icfg.logging = cfg2meta("inference.logging", LoggingConfig())
        icfg.validation_data = cfg2meta("inference.validation_data", DataLoaderParams())
        icfg.prediction_data = cfg2meta("inference.prediction_data", DataLoaderParams())

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self):
        self.logging.configure_wandb( config=to_flat_dict(dataclasses.asdict(self)), resume=False )

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




