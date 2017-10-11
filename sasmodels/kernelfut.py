"""
Python driver for python kernels

Calls the kernel with a vector of $q$ values for a single parameter set.
Polydispersity is supported by looping over different parameter sets and
summing the results.  The interface to :class:`PyModel` matches those for
:class:`kernelcl.GpuModel` and :class:`kerneldll.DllModel`.
"""
from __future__ import division, print_function

import logging

import numpy as np  # type: ignore
from numpy import pi, sin, cos  #type: ignore

from . import details
from .generate import F16, F32, F64
from .kernel import KernelModel, Kernel
from imp import load_source
from .kernelcl import GpuInput
import pyopencl as cl
from pyopencl import mem_flags as mf

try:
    from typing import Union, Callable
except ImportError:
    pass
else:
    DType = Union[None, str, np.dtype]

MODELS_FOLDER = "sasmodels/models"

class FutModel(KernelModel):
    """
    Wrapper for futhark models.
    """
    def __init__(self, model_info, dtype):
        # Make sure Iq and Iqxy are available and vectorized
        #_create_default_functions(model_info)
        self.info = model_info
        self.dtype = dtype

    def make_kernel(self, q_vectors):
        logging.info("creating python kernel " + self.info.name)
        q_input = GpuInput(q_vectors, dtype=self.dtype)
        kernel = self.info.Iqxy if q_input.is_2d else self.info.Iq
        return FutKernel(kernel, self.info, q_input)

    def release(self):
        """
        Free resources associated with the model.
        """
        pass


class FutKernel(Kernel):
    """
    Callable SAS kernel.

    *kernel* is the DllKernel object to call.

    *model_info* is the module information

    *q_input* is the DllInput q vectors at which the kernel should be
    evaluated.

    The resulting call method takes the *pars*, a list of values for
    the fixed parameters to the kernel, and *pd_pars*, a list of (value,weight)
    vectors for the polydisperse parameters.  *cutoff* determines the
    integration limits: any points with combined weight less than *cutoff*
    will not be calculated.

    Call :meth:`release` when done with the kernel instance.
    """
    def __init__(self, kernel, model_info, q_input):
        # type: (callable, ModelInfo, List[np.ndarray]) -> None
        self.dtype = q_input.dtype
        self.info = model_info
        self.q_input = q_input
        self.res = np.empty(q_input.nq, q_input.dtype)
        self.kernel = kernel
        self.dim = '2d' if q_input.is_2d else '1d'
        self.futhark_kernel = None
        self.call_details_buffer = None
        self.values = None

        partable = model_info.parameters
        kernel_parameters = (partable.iqxy_parameters if q_input.is_2d
                             else partable.iq_parameters)
        volume_parameters = partable.form_volume_parameters

        # Create an array to hold the parameter values.  There will be a
        # single array whose values are updated as the calculator goes
        # through the loop.  Arguments to the kernel and volume functions
        # will use views into this vector, relying on the fact that a
        # an array of no dimensions acts like a scalar.
        parameter_vector = np.empty(len(partable.call_parameters)-2, 'd')

        # Create views into the array to hold the arguments
        offset = 0
        kernel_args, volume_args = [], []
        for p in partable.kernel_parameters:
            if p.length == 1:
                # Scalar values are length 1 vectors with no dimensions.
                v = parameter_vector[offset:offset+1].reshape(())
            else:
                # Vector values are simple views.
                v = parameter_vector[offset:offset+p.length]
            offset += p.length
            if p in kernel_parameters:
                kernel_args.append(v)
            if p in volume_parameters:
                volume_args.append(v)

        # Hold on to the parameter vector so we can use it to call kernel later.
        # This may also be required to preserve the views into the vector.
        self._parameter_vector = parameter_vector

        # Generate a closure which calls the kernel with the views into the
        # parameter array.

        model_name = self.info.name
        model_path = self.info.futhark_path
        dtype = self.dtype
        total_path = "%s/%s" % (MODELS_FOLDER, model_path)
        _class = load_source(model_name, total_path)
        self.futhark_kernel = getattr(_class, model_name)(num_groups=4, group_size=32)
        queue = self.futhark_kernel.queue

        if q_input.is_2d:
            qx, qy = q_input.q[:, 0], q_input.q[:, 1]
            qx_flat = np.ascontiguousarray(qx, dtype)
            qy_flat = np.ascontiguousarray(qy, dtype)
            self.q_input.qx = cl.array.to_device(queue, qx_flat)
            self.q_input.qy = cl.array.to_device(queue, qy_flat)
            self._form = getattr(self.futhark_kernel, "kernel_%s_2d"%dtype)

        else:
            self.q_input.q = cl.array.to_device(queue, q_input.q)
            self._form = getattr(self.futhark_kernel, "kernel_%s"%dtype)

        # Generate a closure which calls the form_volume if it exists.
        form_volume = model_info.form_volume
        self._volume = ((lambda: form_volume(*volume_args)) if form_volume
                        else (lambda: 1.0))

        self.real = (np.float32 if dtype == F32
                     else np.float64 if dtype == F64
        else np.float16 if dtype == F16
        else np.float32)  # will never get here, so use np.float32

        ## GET MONO
        parameters = self.info.parameters



        # create call_details, values, is_magnetic
        # call_details, values, is_magnetic = make_kernel_args(calculator, vw_pairs)


    def __call__(self, call_details, values, cutoff, magnetic):
        # type: (CallDetails, np.ndarray, np.ndarray, float, bool) -> np.ndarray
        if magnetic:
            raise NotImplementedError("Magnetism not implemented for pure python models")
        #print("Calling python kernel")
        #call_details.show(values)

        n_pars = len(self._parameter_vector)

        queue = self.futhark_kernel.queue
        if self.call_details_buffer is None:
            self.call_details_buffer = cl.array.to_device(queue, call_details.buffer)

        if self.values is None:
            self.values = cl.array.to_device(queue, values)

        if self.q_input.is_2d:
            args = [
                n_pars,
                call_details.num_active,
                np.uint32(self.q_input.nq),
                call_details.num_eval,
                self.call_details_buffer,
                self.values,
                self.q_input.qx,
                self.q_input.qy,
                self.real(cutoff),
            ]
        else:
            args = [
                n_pars,
                call_details.num_active,
                np.uint32(self.q_input.nq),
                call_details.num_eval,
                self.call_details_buffer,
                self.values,
                self.q_input.q,
                self.real(cutoff),
            ]

        return self._form(*args).get()

    def release(self):
        # type: () -> None
        """
        Free resources associated with the kernel.
        """
        self.q_input.release()
        self.q_input = None

def _loops(parameters, form, form_volume, nq, call_details, values, cutoff):
    # type: (np.ndarray, Callable[[], np.ndarray], Callable[[], float], int, details.CallDetails, np.ndarray, np.ndarray, float) -> None
    ################################################################
    #                                                              #
    #   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   #
    #   !!                                                    !!   #
    #   !!  KEEP THIS CODE CONSISTENT WITH KERNEL_TEMPLATE.C  !!   #
    #   !!                                                    !!   #
    #   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   #
    #                                                              #
    ################################################################
    n_pars = len(parameters)
    parameters[:] = values[2:n_pars+2]
    if call_details.num_active == 0:
        pd_norm = float(form_volume())
        scale = values[0]/(pd_norm if pd_norm != 0.0 else 1.0)
        background = values[1]
        return scale*form() + background

    pd_value = values[2+n_pars:2+n_pars + call_details.num_weights]
    pd_weight = values[2+n_pars + call_details.num_weights:]

    pd_norm = 0.0
    spherical_correction = 1.0
    partial_weight = np.NaN
    weight = np.NaN

    p0_par = call_details.pd_par[0]
    p0_is_theta = (p0_par == call_details.theta_par)
    p0_length = call_details.pd_length[0]
    p0_index = p0_length
    p0_offset = call_details.pd_offset[0]

    pd_par = call_details.pd_par[:call_details.num_active]
    pd_offset = call_details.pd_offset[:call_details.num_active]
    pd_stride = call_details.pd_stride[:call_details.num_active]
    pd_length = call_details.pd_length[:call_details.num_active]

    total = np.zeros(nq, 'd')
    for loop_index in range(call_details.num_eval):
        # update polydispersity parameter values
        if p0_index == p0_length:
            pd_index = (loop_index//pd_stride)%pd_length
            parameters[pd_par] = pd_value[pd_offset+pd_index]
            partial_weight = np.prod(pd_weight[pd_offset+pd_index][1:])
            if call_details.theta_par >= 0:
                cor = sin(pi / 180 * parameters[call_details.theta_par])
                spherical_correction = max(abs(cor), 1e-6)
            p0_index = loop_index%p0_length

        weight = partial_weight * pd_weight[p0_offset + p0_index]
        parameters[p0_par] = pd_value[p0_offset + p0_index]
        if p0_is_theta:
            cor = cos(pi/180 * parameters[p0_par])
            spherical_correction = max(abs(cor), 1e-6)
        p0_index += 1
        if weight > cutoff:
            # Call the scattering function
            # Assume that NaNs are only generated if the parameters are bad;
            # exclude all q for that NaN.  Even better would be to have an
            # INVALID expression like the C models, but that is too expensive.
            Iq = np.asarray(form(), 'd')
            if np.isnan(Iq).any():
                continue

            # update value and norm
            weight *= spherical_correction
            total += weight * Iq
            pd_norm += weight * form_volume()

    scale = values[0]/(pd_norm if pd_norm != 0.0 else 1.0)
    background = values[1]
    return scale*total + background


def _create_default_functions(model_info):
    """
    Autogenerate missing functions, such as Iqxy from Iq.

    This only works for Iqxy when Iq is written in python. :func:`make_source`
    performs a similar role for Iq written in C.  This also vectorizes
    any functions that are not already marked as vectorized.
    """
    _create_vector_Iq(model_info)
    _create_vector_Iqxy(model_info)  # call create_vector_Iq() first


def _create_vector_Iq(model_info):
    """
    Define Iq as a vector function if it exists.
    """
    Iq = model_info.Iq
    if callable(Iq) and not getattr(Iq, 'vectorized', False):
        #print("vectorizing Iq")
        def vector_Iq(q, *args):
            """
            Vectorized 1D kernel.
            """
            return np.array([Iq(qi, *args) for qi in q])
        vector_Iq.vectorized = True
        model_info.Iq = vector_Iq

def _create_vector_Iqxy(model_info):
    """
    Define Iqxy as a vector function if it exists, or default it from Iq().
    """
    Iq, Iqxy = model_info.Iq, model_info.Iqxy
    if callable(Iqxy):
        if not getattr(Iqxy, 'vectorized', False):
            #print("vectorizing Iqxy")
            def vector_Iqxy(qx, qy, *args):
                """
                Vectorized 2D kernel.
                """
                return np.array([Iqxy(qxi, qyi, *args) for qxi, qyi in zip(qx, qy)])
            vector_Iqxy.vectorized = True
            model_info.Iqxy = vector_Iqxy
    elif callable(Iq):
        #print("defaulting Iqxy")
        # Iq is vectorized because create_vector_Iq was already called.
        def default_Iqxy(qx, qy, *args):
            """
            Default 2D kernel.
            """
            return Iq(np.sqrt(qx**2 + qy**2), *args)
        default_Iqxy.vectorized = True
        model_info.Iqxy = default_Iqxy
