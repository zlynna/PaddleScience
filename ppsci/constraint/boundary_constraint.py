"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os.path as osp
import types

import numpy as np
import sympy
from sympy.parsing import sympy_parser as sp_parser

from ppsci import geometry
from ppsci.constraint import base
from ppsci.data import dataset
from ppsci.utils import misc


class BoundaryConstraint(base.Constraint):
    """Class for boundary constraint.

    Args:
        label_expr (Dict[str, sympy.Basic]): Expression of how to compute label.
        label_dict (Dict[str, Union[float, sympy.Basic]]): Value of label.
        geom (Geometry): Geometry which constraint applied on.
        dataloader_cfg (AttrDict): Config of building a dataloader.
        loss (LossBase): Loss object.
        random (str, optional): Random method for sampling points in geometry.
            Defaults to "pseudo".
        criteria (Callable, optional): Criteria for finely define subdomain in geometry.
            Defaults to None.
        evenly (bool, optional): Whether to use envely distribution in sampling.
            Defaults to False.
        weight_dict (Dict[str, Union[float, sympy.Basic]], optional): Weight for label
            if specified. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "BC".
    """

    def __init__(
        self,
        label_expr,
        label_dict,
        geom,
        dataloader_cfg,
        loss,
        random="pseudo",
        criteria=None,
        evenly=False,
        weight_dict=None,
        name="BC",
    ):
        self.label_expr = label_expr
        for label_name, label_expr in self.label_expr.items():
            if isinstance(label_expr, str):
                self.label_expr[label_name] = sp_parser.parse_expr(label_expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())
        # "area" will be kept in "output_dict" for computation.
        if isinstance(geom, geometry.Mesh):
            self.output_keys += ["area"]

        if isinstance(criteria, str):
            criteria = eval(criteria)

        input = geom.sample_boundary(
            dataloader_cfg["sampler"]["batch_size"] * dataloader_cfg["iters_per_epoch"],
            random,
            criteria,
            evenly,
        )
        if "area" in input:
            input["area"] *= dataloader_cfg["iters_per_epoch"]

        label = {}
        for key, value in label_dict.items():
            if isinstance(value, (int, float)):
                label[key] = np.full_like(next(iter(input.values())), float(value))
            elif isinstance(value, sympy.Basic):
                func = sympy.lambdify(
                    sympy.symbols(geom.dim_keys),
                    value,
                    [{"amax": lambda xy, axis: np.maximum(xy[0], xy[1])}, "numpy"],
                )
                label[key] = func(
                    **{k: v for k, v in input.items() if k in geom.dim_keys}
                )
            elif isinstance(value, types.FunctionType):
                func = value
                label[key] = func(input)
                if isinstance(label[key], (int, float)):
                    label[key] = np.full_like(
                        next(iter(input.values())), float(label[key])
                    )
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in geom.dim_keys],
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(**{k: input[k] for k in geom.dim_keys})
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), float(weight[key])
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)
        super().__init__(_dataset, dataloader_cfg, loss, name)


class SupervisedConstraint(base.Constraint):
    """Class for supervised constraint.

    Args:
        label_expr (Dict[str, sympy.Basic]): Expression of how to compute label.
        data_file (Dict[str, Union[float, sympy.Basic]]): Path of data file.
        input_keys (List[str]): List of input keys.
        dataloader_cfg (AttrDict): Config of building a dataloader.
        loss (LossBase): Loss object.
        weight_dict (Dict[str, Union[float, sympy.Basic]], optional): Weight for label
            if specified. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "SupBC".
    """

    def __init__(
        self,
        data_file,
        input_keys,
        label_keys,
        alias_dict,
        dataloader_cfg,
        loss,
        weight_dict=None,
        timestamps=None,
        name="SupBC",
    ):
        self.input_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]
        self.output_keys = [
            alias_dict[key] if key in alias_dict else key for key in label_keys
        ]
        if data_file.endswith(".csv"):
            # load data
            data = misc.load_csv_file(data_file, input_keys + label_keys, alias_dict)
            if "t" not in data and timestamps is None:
                raise ValueError(f"must given time data from t0 or data itself.")
            if timestamps is not None:
                if "t" in data:
                    raw_time_array = data["t"]
                    mask = np.zeros((len(raw_time_array),), "bool")
                    for ti in timestamps:
                        mask |= np.isclose(raw_time_array, ti).flatten()
                    data = misc.convert_to_array(
                        data, self.input_keys + self.output_keys
                    )
                    data = data[mask]
                    data = misc.convert_to_dict(
                        data, self.input_keys + self.output_keys
                    )
                else:
                    data = misc.convert_to_array(
                        data, self.input_keys + self.output_keys
                    )
                    data = misc.combine_array_with_time(data, timestamps)
                    self.input_keys = ["t"] + self.input_keys
                    data = misc.convert_to_dict(
                        data, self.input_keys + self.output_keys
                    )
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}
                self.num_timestamp = len(timestamps)
            else:
                # time already in data and "t" in input_keys
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}
                self.num_timestamp = len(np.unique(data["t"]))

            self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        else:
            raise NotImplementedError(f"Only suppport .csv file now.")

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in self.input_keys],
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(**{k: input[k] for k in self.input_keys})
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), float(weight[key])
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)

        super().__init__(_dataset, dataloader_cfg, loss, name)