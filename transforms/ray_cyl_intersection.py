#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:51:38 2018

@author: bernier2, sfli
"""
import numpy as np
import tensorflow as tf

epsf = np.finfo(float).eps
sqrt_epsf = np.sqrt(epsf)


def unit_vector(invec):
    invec = np.atleast_2d(invec)
    npts, dim = invec.shape

    invec[np.isnan(invec)] = 0.  # supress RuntimeWarning for invaild values

    nrm = np.sqrt(np.sum(invec*invec, axis=1))
    zidx = nrm < epsf
    nrm[zidx] = 1.

    outvec = invec*np.tile(1./nrm, (dim, 1)).T
    outvec[zidx, :] = 0.
    return outvec


def ray_plane(rays, rmat, tvec):
    """
    tvec: Is the normal defining the plane.
    rvec
    """
    nhat = rmat[:, 2].reshape(3, 1)

    rays = np.atleast_2d(rays)  # shape (npts, 3)
    npts = len(rays)

    numerator = np.tile(np.dot(tvec, nhat), npts)
    denominator = np.dot(rays, nhat).flatten()
    can_intersect = denominator < 0

    u = np.tile(numerator[can_intersect] / denominator[can_intersect], (3, 1))

    output = np.nan*np.ones_like(rays)
    output[can_intersect, :] = rays[can_intersect] * u.T

    return np.dot(output - tvec, rmat)[:, :2]


def tf_ray_plane(rays, rmat, tvec):
    """
    In general, the plane equation is defined by
    $$\hat{n} \cdot (\vec{x} - \vec{x}_0)  = 0$$

    where $\vec{x}_0$ defines the origin of rotation for the plane, $\hat{n}$
    defines the plane normal.

    For a given ray, $\vec{r}(t) = \vec{p}_0 + \hat{u} t$

    the ray-plane intersection is defined by

    $$t = \frac{\hat{n} \cdot (\vec{x}_0 - \vec{p}_0)} {\hat{n} \cdot \hat{u}} $$


    To parameterize the detector plane, we will give the origin point a
    translation vector.

    This gives us

    $$ \mathbf{R} \hat{n} \cdot (\vec{x} - (\vec{x}_0 + \vec{d}))  = 0$$

    where $\vec{d}$ is the translation vector and $\mathbf{R}$ is the rotation
    of the detector about its origin.


    For this function, we'll take \vec{x_0} to be at the origin. We will define
    the translation and rotation as `tvec` and `rmat`


    :param tvec: Translation vector for the origin of the plane.
    :param rmat: Rotation of the detector plane about its origin.
    :param rays: Ray vector with $\vec{p}_0$ taken to be coordinate origin.

    :return: The list of x, y intersection location of the plane.

    As a convention, for tvec = [0, 0, 0], and R = np.eye(3), the detector
    plane is located at the origin, with its normal pointing in the [0, 0, 1]
    direction.

    Similarly, by convention, the origin of the ray, $\vec{p}_0$ is taken to
    be the origin as well.

    """

    # TODO(Frankie): Static type check to ensure that rmat, tvec, and ray
    # are already type, `tf.tensor`.

    # Detector normal, rotated about its origin.
    # Detector normal, rotated about its origin.
    nhat = rmat[:, 2]

    numerator = tf.broadcast_to(tf.tensordot(tvec, nhat, axes=1), [tf.shape(rays)[0], 1])
    denominator = tf.tensordot(rays, nhat, axes=1)
    tmp = numerator / tf.reshape(denominator, [tf.shape(rays)[0], 1])

    # Using `d_censor` to replace the two lines below
    # can_intersect = denominator < 0
    # output[can_intersect, :] = rays[can_intersect] * u.T
    d_censor = tf.scan(lambda a, x: 1.0 if x < 0 else np.nan, denominator)
    tmp = tmp * tf.reshape(d_censor, [tf.shape(rays)[0], 1])
    u = tf.broadcast_to(tmp, [tf.shape(rays)[0], 3])
    output = rays * u

    # We only take the x, y, the detector plane coordinates.
    return tf.tensordot(output - tvec, rmat, axes=[[1],[1]])[:, :2]



def ray_cylinder(rays, radius, rmat, tvec):
    """
    return points P in ref frame (origin [0, 0, 0]) associated with rays n

    parameterize ray as p = u*d for scalar u and unit vector d

    in ref frame, vector to point on cylinder is l = p - t
    """
    rays = np.atleast_2d(rays)  # shape (npts, 3)
    npts = len(rays)

    output = np.nan * np.ones_like(rays)

    r_0 = rmat[:, 0]
    r_2 = rmat[:, 2]

    a = np.dot(rays, r_0)
    b = np.tile(np.dot(tvec, r_0), npts)
    c = np.dot(rays, r_2)
    d = np.tile(np.dot(tvec, r_2), npts)

    aq = a**2 + c**2
    bq = -2.*(a*b + c*d)
    cq = b**2 + d**2 - radius**2

    discriminant = bq**2 - 4.*aq*cq
    can_intersect = discriminant >= sqrt_epsf
    discriminant[~can_intersect] = np.nan
    aq[aq == 0] = 1.    # nans in discriminant will persist

    # the two roots for the ray vector scaling
    root1 = 0.5*(-bq + np.sqrt(discriminant))/aq
    root2 = -0.5*(bq + np.sqrt(discriminant))/aq
    root1[np.isnan(root1)] = 0.
    root2[np.isnan(root2)] = 0.
    root1[root1 <= 0] = np.nan
    root2[root2 <= 0] = np.nan

    p_both = np.vstack([np.tile(root1, (3, 1)).T*rays,
                        np.tile(root2, (3, 1)).T*rays])
    l_both = p_both - tvec

    rdp = np.sum(unit_vector(p_both)*unit_vector(l_both), axis=1)
    rdp[np.isnan(rdp)] = 0.

    idx_arr = np.tile(range(npts), (2, 1))
    hits_inner_surf = (rdp > 0).reshape(2, npts)
    if np.any(np.sum(hits_inner_surf, axis=0) > 1):
        raise RuntimeError("something is wrong...")
    output[idx_arr[hits_inner_surf], :] = np.dot(
            l_both[hits_inner_surf.flatten(), :],
            rmat)
    return output

