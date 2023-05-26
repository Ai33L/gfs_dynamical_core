import pytest
import abc
import os
import sys
from glob import glob
import xarray as xr
import numpy as np
import logging
from gfs_dynamical_core import (GFSDynamicalCore)
from sympl import (
    Stepper, TendencyStepper, TimeDifferencingWrapper,
    ImplicitTendencyComponent, UpdateFrequencyWrapper, DataArray,
    TendencyComponent, AdamsBashforth
)
from sympl._core.tracers import reset_tracers, reset_packers
from datetime import datetime, timedelta
import climt
os.environ['NUMBA_DISABLE_JIT'] = '1'

vertical_dimension_names = [
    'interface_levels', 'mid_levels', 'full_levels']

cache_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'cached_component_output')


def cache_dictionary(dictionary, filename):
    dataset = xr.Dataset(dictionary)
    dataset.to_netcdf(filename, engine='scipy')


def load_dictionary(filename):
    dataset = xr.open_dataset(filename, engine='scipy')
    return_dict = dict(dataset.data_vars)
    return_dict.update(dataset.coords)
    return return_dict


def state_3d_to_1d(state):
    return_state = {}
    for name, value in state.items():
        if name == 'time':
            return_state[name] = value
        else:
            dim_list = []
            for i, dim in enumerate(value.dims):
                if dim in vertical_dimension_names:
                    dim_list.append(slice(0, value.shape[i]))
                else:
                    dim_list.append(0)
            return_state[name] = value[tuple(dim_list)]
    return return_state


def transpose_state(state, dims=None):
    return_state = {}
    for name, value in state.items():
        if name == 'time':
            return_state[name] = state[name]
        else:
            if dims is None:
                return_state[name] = state[name].transpose()
            else:
                return_state[name] = state[name].transpose(*dims)
    return return_state


def call_with_timestep_if_needed(
        component, state, timestep=timedelta(seconds=10.)):
    np.random.seed(0)
    if isinstance(component, (Stepper, TendencyStepper, ImplicitTendencyComponent)):
        return component(state, timestep=timestep)
    else:
        return component(state)


class ComponentBase(object):

    def setUp(self):
        reset_tracers()
        reset_packers()
        super(ComponentBase, self).setUp()

    @abc.abstractmethod
    def get_component_instance(self):
        pass

    def get_cache_filename(self, descriptor, i):
        return '{}-{}-{}.cache'.format(self.__class__.__name__, descriptor, i)

    def get_cached_output(self, descriptor):
        cache_filename_list = sorted(glob(
            os.path.join(
                cache_folder,
                self.get_cache_filename(descriptor, '*'))))
        if len(cache_filename_list) > 0:
            return_list = []
            for filename in cache_filename_list:
                return_list.append(load_dictionary(filename))
            if len(return_list) > 1:
                return tuple(return_list)
            elif len(return_list) == 1:
                return return_list[0]
        else:
            return None

    def cache_output(self, output, descriptor):
        if not isinstance(output, tuple):
            output = (output,)
        for i in range(len(output)):
            cache_filename = os.path.join(
                cache_folder, self.get_cache_filename(descriptor, i))
            cache_dictionary(output[i], cache_filename)

    def assert_valid_output(self, output):
        if isinstance(output, dict):
            output = [output]
        for i, out_dict in enumerate(output):
            for name, value in out_dict.items():
                try:
                    if name != 'time' and np.any(np.isnan(value)):
                        raise AssertionError(
                            'NaN produced in output {} from dict {}'.format(name, i))
                except TypeError:  # raised if cannot run isnan on dtype of value
                    pass


class ComponentBaseColumn(ComponentBase):

    def get_1d_input_state(self, component=None):
        if component is None:
            component = self.get_component_instance()
        return climt.get_default_state(
            [component], grid_state=climt.get_grid(nx=None, ny=None, nz=30))

    def test_column_output_matches_cached_output(self):
        state = self.get_1d_input_state()
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output('column')
        if cached_output is None:
            self.cache_output(output, 'column')
            raise AssertionError(
                'Failed due to no cached output, cached current output.')
        else:
            compare_outputs(output, cached_output)

    def test_no_nans_in_column_output(self):
        state = self.get_1d_input_state()
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        self.assert_valid_output(output)

    def test_column_stepping_output_matches_cached_output(self):
        component = self.get_component_instance()
        if isinstance(component, (TendencyComponent, ImplicitTendencyComponent)):
            component = AdamsBashforth(self.get_component_instance())
            state = self.get_1d_input_state(component)
            output = call_with_timestep_if_needed(component, state)
            cached_output = self.get_cached_output('column_stepping')
            if cached_output is None:
                self.cache_output(output, 'column_stepping')
                raise AssertionError(
                    'Failed due to no cached output, cached current output.')
            else:
                compare_outputs(output, cached_output)


class ComponentBase3D(ComponentBase):

    def get_3d_input_state(self, component=None):
        if component is None:
            component = self.get_component_instance()
        return climt.get_default_state(
            [component], grid_state=climt.get_grid(nx=32, ny=16, nz=28))

    def test_3d_output_matches_cached_output(self):
        state = self.get_3d_input_state()
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output('3d')
        if cached_output is None:
            self.cache_output(output, '3d')
            raise AssertionError(
                'Failed due to no cached output, cached current output.')
        else:
            compare_outputs(output, cached_output)

    def test_3d_stepping_output_matches_cached_output(self):
        component = self.get_component_instance()
        if isinstance(component, (TendencyComponent, ImplicitTendencyComponent)):
            component = AdamsBashforth(component)
            state = self.get_3d_input_state(component)
            output = call_with_timestep_if_needed(component, state)
            cached_output = self.get_cached_output('3d_stepping')
            if cached_output is None:
                self.cache_output(output, '3d_stepping')
                raise AssertionError(
                    'Failed due to no cached output, cached current output.')
            else:
                compare_outputs(output, cached_output)

    def test_no_nans_in_3D_output(self):
        state = self.get_3d_input_state()
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        self.assert_valid_output(output)

    def test_reversed_state_gives_same_output(self):
        state = self.get_3d_input_state()
        for name, value in state.items():
            if isinstance(value, (timedelta, datetime)):
                pass
            elif len(value.dims) == 3:
                state[name] = value.transpose(value.dims[2], value.dims[1], value.dims[0])
            elif len(value.dims) == 2:
                state[name] = value.transpose(value.dims[1], value.dims[0])
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output('3d')
        if cached_output is None:
            raise AssertionError(
                'Failed due to no cached output.')
        else:
            compare_outputs(output, cached_output)

    def test_transposed_state_gives_same_output(self):
        state = self.get_3d_input_state()
        for name, value in state.items():
            if isinstance(value, (timedelta, datetime)):
                pass
            elif len(value.dims) == 3:
                state[name] = value.transpose(value.dims[2], value.dims[0], value.dims[1])
            elif len(value.dims) == 2:
                state[name] = value.transpose(value.dims[1], value.dims[0])
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output('3d')
        if cached_output is None:
            raise AssertionError(
                'Failed due to no cached output.')
        else:
            compare_outputs(output, cached_output)


def compare_outputs(current, cached):
    if isinstance(current, tuple) and isinstance(cached, tuple):
        for i in range(len(current)):
            compare_one_state_pair(current[i], cached[i])
    elif (not isinstance(current, tuple)) and (not isinstance(cached, tuple)):
        compare_one_state_pair(current, cached)
    else:
        raise AssertionError('Different number of dicts returned than cached.')


def compare_one_state_pair(current, cached):
    for key in current.keys():
        if key == 'time':
            assert key in cached.keys()
        else:
            try:
                if not np.all(current[key] == cached[key]):
                    assert np.all(np.isclose(current[key] - cached[key], 0.))
                for attr in current[key].attrs:
                    assert current[key].attrs[attr] == cached[key].attrs[attr]
                for attr in cached[key].attrs:
                    assert attr in current[key].attrs
                assert set(current[key].dims) == set(cached[key].dims)
            except AssertionError as err:
                raise AssertionError('Error for {}: {}'.format(key, err))
    for key in cached.keys():
        assert key in current.keys()


# @pytest.mark.skip("fails on CI, no idea why")
class TestGFSDycore(ComponentBase3D):

    def get_component_instance(self):
        return GFSDynamicalCore()
