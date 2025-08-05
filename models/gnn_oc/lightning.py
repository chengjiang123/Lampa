import importlib
import math
import os
from pathlib import Path
from typing import Any, Literal, Union

import pytorch_lightning
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor, nn
from torchmetrics import Metric
from tqdm import tqdm

# std


import logging

import colorlog

LOG_DEFAULT_LEVEL = logging.DEBUG


def get_logger(name="gnn-tracking", level=LOG_DEFAULT_LEVEL):
    """Sets up global logger."""
    _log = colorlog.getLogger(name)

    if _log.handlers:
        # the logger already has handlers attached to it, even though
        # we didn't add it ==> logging.get_logger got us an existing
        # logger ==> we don't need to do anything
        return _log

    _log.setLevel(level)

    sh = colorlog.StreamHandler()
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    #  This is not the same as just setting name="" in the fct arguments.
    #  This would set the root logger to debug mode, which for example causes
    #  the matplotlib font manager (which uses the root logger) to throw lots of
    #  messages. Here, we want to keep our named logger, but just drop the
    #  name.
    name_incl = "" if name == "gnn-tracking" else f" {name}"
    formatter = colorlog.ColoredFormatter(
        f"%(log_color)s[%(asctime)s{name_incl}] %(levelname)s: %(message)s",
        log_colors=log_colors,
        datefmt="%H:%M:%S",
    )
    sh.setFormatter(formatter)
    # Controlled by overall logger level
    sh.setLevel(logging.DEBUG)

    _log.addHandler(sh)

    return _log


logger = get_logger()


def save_sub_hyperparameters(
    self: HyperparametersMixin,
    key: str,
    obj: Union[HyperparametersMixin, dict],
    errors: Literal["warn", "raise"] = "warn",
) -> None:
    """Take hyperparameters from `obj` and save them to `self` under the
    key `key`.

    Args:
        self: The object to save the hyperparameters to.
        key: The key under which to save the hyperparameters.
        obj: The object to take the hyperparameters from.
        errors: Whether to raise an error or just warn
    """
    if not hasattr(obj, "hparams"):
        msg = (
            "Can't save hyperparameters from object of type %s. Make sure to "
            "inherit from HyperparametersMixin."
        )
        if errors == "warn":
            logger.warning(msg, type(obj))
            return
        if errors == "raise":
            _ = msg % type(obj)
            raise ValueError(_)
        _ = f"Unknown value for `errors`: {errors}"
        raise ValueError(_)

    assert key not in self.hparams
    if isinstance(obj, dict):
        logger.warning("SSH got dict %s. That's unexpected.", obj)
        sub_hparams = obj
    else:
        sub_hparams = {
            "class_path": obj.__class__.__module__ + "." + obj.__class__.__name__,
            "init_args": dict(obj.hparams),
        }
    self.save_hyperparameters({key: sub_hparams})


def load_obj_from_hparams(hparams: dict[str, Any], key: str = "") -> Any:
    """Load object from hyperparameters."""
    if key:
        hparams = hparams[key]
    return get_object_from_path(hparams["class_path"], hparams["init_args"])


def obj_from_or_to_hparams(self: HyperparametersMixin, key: str, obj: Any) -> Any:
    """Used to support initializing python objects from hyperparameters:
    If `obj` is a python object other than a dictionary, its hyperparameters are
    saved (its class path and init args) to `self.hparams[key]`.
    If `obj` is instead a dictionary, its assumed that we have to restore an object
    based on this information.
    """
    if isinstance(obj, dict) and "class_path" in obj and "init_args" in obj:
        self.save_hyperparameters({key: obj})
        return load_obj_from_hparams(obj)
    if isinstance(obj, (int, float, str, bool, list, tuple, dict)) or obj is None:
        self.save_hyperparameters({key: obj})
        return obj
    save_sub_hyperparameters(self=self, key=key, obj=obj)  # type: ignore
    return obj


def get_object_from_path(path: str, init_args: Union[dict[str, Any], None] = None) -> Any:
    """Get object from path (string) to its code location."""
    module_name, _, class_name = path.rpartition(".")
    logger.debug("Getting class %s from module %s", class_name, module_name)
    if not module_name:
        msg = "Please specify the full import path"
        raise ValueError(msg)
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    if init_args is not None:
        return obj(**init_args)
    return obj
