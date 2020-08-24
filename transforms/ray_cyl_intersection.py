#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:51:38 2018

@author: bernier2
"""
import numpy as np

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
    """
    nhat = rmat[:, 2].reshape(3, 1)

    rays = np.atleast_2d(rays)  # shape (npts, 3)
    npts = len(rays)

    output = np.nan*np.ones_like(rays)

    numerator = np.tile(np.dot(tvec, nhat), npts)
    denominator = np.dot(rays, nhat).flatten()

    can_intersect = denominator < 0
    u = np.tile(numerator[can_intersect]/denominator[can_intersect], (3, 1))
    output[can_intersect, :] = rays[can_intersect] * u.T

    return np.dot(output - tvec, rmat)[:, :2]


def ray_cylinder(rays, radius, rmat, tvec):
    """
    return points P in ref frame (origin [0, 0, 0]) associated with rays n

    parameterize ray as p = u*d for scalar u and unit vector d

    in ref frame, vector to point on cylinder is l = p - t
    """
    rays = np.atleast_2d(rays)  # shape (npts, 3)
    npts = len(rays)

    output = np.nan*np.ones_like(rays)

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
