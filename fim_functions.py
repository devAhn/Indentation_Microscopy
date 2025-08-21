# cmd shift p -> interpreter

import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import cv2


    # use_positive_solution
    # False tracks the First Collision Point
    # True tracks the second Collision Point
def propagate_to_sphere(r_rays, u_rays, radius, use_positive_solution=False):
    # propagate rays (defined by r - position, u - direction) to surface of sphere
    # radius is radius of glass sphere
    # positive vs negative solution determines whether you want to intersect with the top vs bottom of sphere
    
    # sphere is centered at the origin;
    u_dot_r = torch.einsum('ij,ij->i', u_rays, r_rays)
    discriminant = radius ** 2 - torch.linalg.norm(r_rays, axis=1) ** 2 + u_dot_r ** 2
    # missed = discriminant[:, None] < 0  # missed the sphere; will yield nan in the next line:
    if use_positive_solution:
        dist_to_sphere = - u_dot_r + torch.sqrt(
            torch.maximum(torch.abs(discriminant), torch.tensor(1e-7)))  # take pos solution;
    else:
        dist_to_sphere = - u_dot_r - torch.sqrt(
            torch.maximum(torch.abs(discriminant), torch.tensor(1e-7)))  # take neg solution;
    # take the abs value to avoid nan, which torch.where doesn't like;
    dist_to_sphere = dist_to_sphere[:, None]
    r_at_sphere = r_rays + u_rays * dist_to_sphere  # position at dome
    return r_at_sphere, dist_to_sphere, discriminant[:, None]


def propagate_to_ellipsoid(r_rays, u_rays, A_ellipse, use_positive_solution=False):
    # same as propagate_to_sphere, but now A_ellipse is a 3x3 positive definite matrix describing an ellipsoid
    # (centered at the origin)

    # discriminant is a little more complicated:
    # u_rays and r_rays are of shape _, 3
    a = torch.einsum('bi,ij,bj->b', u_rays, A_ellipse, u_rays)
    b = 2 * torch.einsum('bi,ij,bj->b', r_rays, A_ellipse, u_rays)
    c = torch.einsum('bi,ij,bj->b', r_rays, A_ellipse, r_rays) - 1
    discriminant = b ** 2 - 4 * a * c
    if use_positive_solution:
        dist_to_sphere = (- b + torch.sqrt(
            torch.maximum(torch.abs(discriminant), torch.tensor(1e-7))))/2/a  # take pos solution;
    else:
        dist_to_sphere = (- b - torch.sqrt(
            torch.maximum(torch.abs(discriminant), torch.tensor(1e-7))))/2/a  # take neg solution;
    # take the abs value to avoid nan, which torch.where doesn't like;
    dist_to_sphere = dist_to_sphere[:, None]
    r_at_sphere = r_rays + u_rays * dist_to_sphere  # position at dome
    return r_at_sphere, dist_to_sphere, discriminant[:, None]


def refract_snell(u_rays, normals, ri_ratio):
    # determine new ray directions given input directions, u, and the corresponding surface normals;
    # ri_ratio: ri_incident/ri_excident
    _n_dot_u = - torch.einsum('ij,ij->i', normals, u_rays)  # negative n dot u;
    sqrt_arg = 1 - ri_ratio ** 2 * (1 - _n_dot_u ** 2)
    refract_error = sqrt_arg[:, None] < 0  # e.g., TIR
    u_out = ri_ratio * u_rays + (ri_ratio * _n_dot_u -
                                 torch.sqrt(torch.maximum(torch.abs(sqrt_arg), torch.tensor(1e-7))))[:, None] * normals
    # take the abs value to avoid nan, which torch.where doesn't like;
    return u_out, refract_error


def ray_intersection_point(r_rays, u_rays, mask=None, force_full_rank=False):
    # r, u - positions and orientations of rays;
    # each are of shape _, _, 3; the last dimension is the 3 coordinates, second to last is the ray dimenison; all
    # other dimensions are for multiplexing;
    # mask: if supplied, then use it to weight different rays; the purpose of using mask here rather than boolean
    # indexing the inputs r and u is so the tensor shapes will always be the same (to allow vmap). Should be 0 or 1,
    # since I don't divide out by the sum of the weights.
    # force_full_rank: pytorch's lstsq function requires A to be full rank (https://github.com/pytorch/pytorch/issues/117122)

    # compute current intersection point, whether you use it or not, for monitoring:
    uxx = u_rays[..., 0] ** 2 - 1
    uyy = u_rays[..., 1] ** 2 - 1
    uzz = u_rays[..., 2] ** 2 - 1
    uxy = u_rays[..., 0] * u_rays[..., 1]
    uyz = u_rays[..., 1] * u_rays[..., 2]
    uxz = u_rays[..., 0] * u_rays[..., 2]

    if mask is not None:
        uxx = uxx * mask
        uyy = uyy * mask
        uzz = uzz * mask
        uxy = uxy * mask
        uyz = uyz * mask
        uxz = uxz * mask

    C = torch.stack([torch.sum(uxx * r_rays[..., 0] + uxy * r_rays[..., 1] + uxz * r_rays[..., 2], dim=-1),
                     torch.sum(uxy * r_rays[..., 0] + uyy * r_rays[..., 1] + uyz * r_rays[..., 2], dim=-1),
                     torch.sum(uxz * r_rays[..., 0] + uyz * r_rays[..., 1] + uzz * r_rays[..., 2], dim=-1)], dim=-1)

    M = torch.stack([torch.stack([torch.sum(uxx, dim=-1), torch.sum(uxy, dim=-1), torch.sum(uxz, dim=-1)], dim=0),
                     torch.stack([torch.sum(uxy, dim=-1), torch.sum(uyy, dim=-1), torch.sum(uyz, dim=-1)], dim=0),
                     torch.stack([torch.sum(uxz, dim=-1), torch.sum(uyz, dim=-1), torch.sum(uzz, dim=-1)], dim=0)],
                    dim=0)

    M = torch.permute(M, (2, 0, 1))

    if force_full_rank:
        M = M + torch.eye(3).to(M.device)[None] * 1e-7 * torch.rand(1).to(M.device)

    if mask is None:
        R = torch.linalg.solve(M, C[..., None])  # the best intersection point;
    else:
        R, _, _, _ = torch.linalg.lstsq(M, C[..., None])  # this one doesn't raise an error if M is singular

    R = R[..., 0]  # remove singleton dim
    
    # compute sum of square residuals; for some reason, torch.linal.lstsq doesn't do this for square matricies:
    # residuals = C - torch.einsum('kij,kj->ki', M, R)
    # SSR = torch.sum(residuals ** 2, dim=-1)

    # this might be better -- it's the actual mean square distance of closest approaches:
    closest_approaches = r_rays + torch.einsum('kij,kij->ki', R[None] - r_rays, u_rays)[:,:,None]*u_rays - R[None]
    MSR = torch.mean(closest_approaches ** 2, dim=(1,2))*3  # dim 1 is all the rays in the ray bundle, dim 2 is the xyz coords (hence *3)
    # and dim0 corresponds to ray bundles for different foci, if desired
    
    return R, MSR  




def propagate_static_shapes(delta_r, r_rays, u_rays, r_sphere, d_to_focus, RI_sphere=1.45, RI_water=1.33,
                            force_full_rank=True, model_ellipsoid=False):
    # propagate ray fan, shifted by delta_r relative to sphere
    # i.e., delta_r shifts the focus of the ray bundle to simulate scanning and stage z translation
    # static shapes: no boolean indexing, so that shapes are statically known -- allows use of vmap and batching in general
    # instead of selectively propagating rays, propagate all, but keep track of which rays are "good"
    # model_ellipsoid: if true, then model an ellipsoid rather than sphere. Also, r_sphere should be the A matrix.

    # starting ray:
    r0 = r_rays + delta_r[None, :] #staring point 
    u0 = u_rays # direction 

    # propagate to first spherical surface
    if model_ellipsoid:
        r_at_sphere1, dist_to_sphere1, discriminant1 = propagate_to_ellipsoid(r0, u0, r_sphere, False)
    else:
        r_at_sphere1, dist_to_sphere1, discriminant1 = propagate_to_sphere(r0, u0, r_sphere, False)
    hit_sphere1 = discriminant1[:, 0] > 0
    u_at_sphere1 = u0
    dist_to_sphere1 = dist_to_sphere1[:, 0]

    # refract at first surface
    normals1 = torch.nn.functional.normalize(r_at_sphere1, dim=1)
    u_at_sphere1, refract_errors1 = refract_snell(u_at_sphere1, normals1, RI_water / RI_sphere)

    # propagate to second spherical surface
    if model_ellipsoid:
        r_at_sphere2, dist_to_sphere2, discriminant2 = propagate_to_ellipsoid(r_at_sphere1, u_at_sphere1, r_sphere, True)
    else:
        r_at_sphere2, dist_to_sphere2, discriminant2 = propagate_to_sphere(r_at_sphere1, u_at_sphere1, r_sphere, True)
    dist_to_sphere2 = dist_to_sphere2[:, 0]

    # refract at second surface
    normals2 = torch.nn.functional.normalize(-r_at_sphere2, dim=1)
    u_at_sphere2, refract_errors2 = refract_snell(u_at_sphere1, normals2, RI_sphere / RI_water)
    hit_sphere2 = discriminant2[:, 0] > 0  # I think this should always be true
    ##
    #propagete to flat plane - 1.5
    ##
    r_focus, MSR = ray_intersection_point(r_at_sphere2[None], u_at_sphere2[None],
                                          mask=hit_sphere1.float() * hit_sphere2.float(), force_full_rank=force_full_rank)
    # record full trajectory for rays that go through sphere:
    r_trajectory_sphere = torch.stack([r0,
                                       r_at_sphere1,
                                       r_at_sphere2,
                                       r_at_sphere2 + u_at_sphere2 * (d_to_focus - dist_to_sphere1 - dist_to_sphere2)[:,
                                                                     None]
                                       ], axis=-1)

    r_focus = r_focus.squeeze()
    discriminant1 = discriminant1.squeeze()
    discriminant2 = discriminant2.squeeze()

    return r0, u0, r_at_sphere1, r_at_sphere2, u_at_sphere1, u_at_sphere2, r_trajectory_sphere, discriminant1, discriminant2, r_focus, hit_sphere2, MSR

def axis_angle_rotmat(axis, angle):
    # return 3D rotation matrix given axis and angle;
    # axis is of shape (3) and angle is a single number;

    axis_unit = torch.nn.functional.normalize(axis,dim=0)  # convert to unit vector
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    ux = axis_unit[0]
    uy = axis_unit[1]
    uz = axis_unit[2]

    r00 = cos + ux ** 2 * (1 - cos)
    r01 = ux * uy * (1 - cos) - uz * sin
    r02 = ux * uz * (1 - cos) + uy * sin
    r10 = ux * uy * (1 - cos) + uz * sin
    r11 = cos + uy ** 2 * (1 - cos)
    r12 = uy * uz * (1 - cos) - ux * sin
    r20 = ux * uz * (1 - cos) - uy * sin
    r21 = uy * uz * (1 - cos) + ux * sin
    r22 = cos + uz ** 2 * (1 - cos)

    rotmat = torch.stack([torch.stack([r00, r01, r02]),
                          torch.stack([r10, r11, r12]),
                          torch.stack([r20, r21, r22])])
    return rotmat