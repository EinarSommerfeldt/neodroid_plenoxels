// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>

namespace {
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

const int TRACE_RAY_BG_CUDA_THREADS = 128;
const int MIN_BG_BLOCKS_PER_SM = 8;
typedef cub::WarpReduce<float> WarpReducef;

//CUSTOM
const int DISTLOSS_RAY_CUDA_THREADS = 32;

namespace device {

/*
Traces ray and saves densities and associated normalized ray positions
*/
__device__ __inline__ int trace_ray_distloss(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        const float& __restrict__ ray_length,
        float* __restrict__ weights,
        float* __restrict__ midpoint_distances,
        float* __restrict__ intervals) {

    if (ray.tmin > ray.tmax) {
        return;
    }

    float t = ray.tmin;
    float t_old = t;
    int i = 0;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]); // Compute x * y + z as a single operation.
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray, //Compute the amount to skip for negative values
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one( 
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        
        if (sigma > opt.sigma_thresh) {
            weights[i] = sigma;
            if (i > 0) {
                intervals[i-1] = (t - t_old)/ray_length; // s_{i+1} - s_i
                midpoint_distances[i-1] = ((t + t_old)-2*ray.tmin)/(ray_length * 2); // (s_{i+1} + s_i)/2
            }
            
            i++;
        }
        t_old  = t;
        t += opt.step_size;
    }
    //assume last interval is step_size wide
    intervals[i-1] = (t - t_old)/ray_length; // s_{N} + step_size - s_N
    midpoint_distances[i-1] = ((t + t_old)-2*ray.tmin)/(ray_length * 2); // (s_{N} + step_size + s_N)/2
    return i;
}


__device__ __inline__ void trace_ray_backward_distloss(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_arr,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        PackedGridOutputGrads& __restrict__ grads
){
    if (ray.tmin > ray.tmax) return;
    float t = ray.tmin;
    int i = 0;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);            // t * ray.dir + ray.origin
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f); //ray in [0, grid.size)
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);

        if (sigma > opt.sigma_thresh) {

            trilerp_backward_cuvol_one_density( //update grads of all contributing voxels (density).
                    grid.links,
                    grads.grad_density_out,
                    grads.mask_out,
                    grid.stride_x,
                    grid.size[2],
                    ray.l, ray.pos, grad_arr[i]);
            i++;
        }
        t += opt.step_size;
    }
}

// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out, // ray color? (E)
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim; //grid.basis_dim = 9
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]); // Compute x * y + z as a single operation.
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray, //Compute the amount to skip for negative values
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one( 
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;

        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one( //?
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id); //Get coefficients for each lane_id
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness;
    }
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        out[lane_colorgrp] = outv;
    }
}

__device__ __inline__ void trace_ray_expected_term(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > opt.sigma_thresh) {
            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            outv += weight * (t / opt.step_size) * ray.world_step;
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    *out = outv;
}

// From Dex-NeRF
__device__ __inline__ void trace_ray_sigma_thresh(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float sigma_thresh,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    float t = ray.tmin;
    *out = 0.f;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > sigma_thresh) {
            *out = (t / opt.step_size) * ray.world_step;
            break;
        }
        t += opt.step_size;
    }
}

__device__ __inline__ void trace_ray_cuvol_backward(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_output,  // residuals [3]
        const float* __restrict__ color_cache,  // color_cache + ray_id * 3
        SingleRaySpec& __restrict__ ray,        // ray_spec[ray_blk_id]
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,  // sphfunc_val[ray_blk_id]
        float* __restrict__ grad_sphfunc_val,   // grad_sphfunc_val[ray_blk_id]
        WarpReducef::TempStorage& __restrict__ temp_storage,    //temp_storage[ray_blk_id]
        float log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        PackedGridOutputGrads& __restrict__ grads,
        float* __restrict__ accum_out,
        float* __restrict__ log_transmit_out
        ) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim)); // mask for first ray of SH coefficent

    float accum = fmaf(color_cache[0], grad_output[0],          //c[0]*g[0] + c[1]*g[1] + c[2]*g[2]
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));   //L_recon kinda?

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }

    if (beta_loss > 0.f) { //L_beta
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp]; // residual for one color

    float log_transmit = 0.f;

    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);            // t * ray.dir + ray.origin
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f); //ray in [0, grid.size)
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one( //trilerp sh coefficients
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id]; //Calc color

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt)); // estimated color weight
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f; // estimated color
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total; //lane_color_total > 0
            total_color *= gout; // c_est * res | Clamp to [+0, infty) (orig) 
            //grid.basis_dim = 9
            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim); 
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;
            //grid.sh_data_dim = sh_data.size(1)
            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim); //mask = all?
            const float grad_common = weight * color_in_01 * gout; //disable grad if total_color < 0
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            if (grid.basis_type != BASIS_TYPE_SH) { // FALSE
                float curr_grad_sphfunc = lane_color * grad_common;
                const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, 2 * grid.basis_dim);
                curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, grid.basis_dim);
                curr_grad_sphfunc += curr_grad_up2;
                if (lane_id < grid.basis_dim) {
                    grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }
            trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out, //update grads of all contributing voxels (SH).
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one_density( //update grads of all contributing voxels (density).
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            // Cancel beta loss out in case of background
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) { *log_transmit_out = log_transmit; }
        // printf("accum_end_fg=%f\n", accum);
        // printf("log_transmit_fg=%f\n", log_transmit);
    }
}


__device__ __inline__ void render_background_forward(
            const PackedSparseGridSpec& __restrict__ grid,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float* __restrict__ out
        ) {

    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    // csi.intersect(inner_radius, &t_last);

    float outv[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < n_steps; ++i) {
        // Between 1 and infty
        float r = n_steps / (n_steps - i - 0.5);
        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }
        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }

        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);

        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }
        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            const float pcnt = (invr_last - invr_mid) * ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                // Not efficient
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0;  // Scale by SH DC factor to help normalize lrs
                outv[i] += weight * fmaxf(color + 0.5f, 0.f);  // Clamp to [+0, infty)
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
    }
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        out[i] += outv[i] + _EXP(log_transmit) * opt.background_brightness;
    }
}

__device__ __inline__ void render_background_backward(
            const PackedSparseGridSpec& __restrict__ grid,
            const float* __restrict__ grad_output,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float accum,
            float sparsity_loss,
            PackedGridOutputGrads& __restrict__ grads
        ) {
    // printf("accum_init=%f\n", accum);
    // printf("log_transmit_init=%f\n", log_transmit);
    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    // csi.intersect(inner_radius, &t_last);
    for (int i = 0; i < n_steps; ++i) {
        float r = n_steps / (n_steps - i - 0.5);

        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }

        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }


        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);
        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }

        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            float total_color = 0.f;
            const float pcnt = ray.world_step * (invr_last - invr_mid) * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            for (int i = 0; i < 3; ++i) {
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0 + 0.5f;  // Scale by SH DC factor to help normalize lrs

                total_color += fmaxf(color, 0.f) * grad_output[i];
                if (color > 0.f) {
                    const float curr_grad_color = C0 * weight * grad_output[i];
                    trilerp_backward_bg_one(
                            grid.background_links,
                            grads.grad_background_out,
                            nullptr,
                            grid.background_reso,
                            grid.background_nlayers,
                            4,
                            ray.l,
                            ray.pos,
                            curr_grad_color,
                            i);
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (invr_last - invr_mid) * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }

            trilerp_backward_bg_one(
                    grid.background_links,
                    grads.grad_background_out,
                    grads.mask_background_out,
                    grid.background_reso,
                    grid.background_nlayers,
                    4,
                    ray.l,
                    ray.pos,
                    curr_grad_sigma,
                    3);

            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
    }
}

// adapted from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
void distloss_forward_pass(
        float* __restrict__ weights,
        float* __restrict__ midpoint_distances,
        float* __restrict__ wm,
        float* __restrict__ w_prefix,
        float* __restrict__ wm_prefix,
        float& __restrict__ w_total,
        float& __restrict__ wm_total,
        int& __restrict__ total_steps) {

    wm[0] = weights[0]*midpoint_distances[0];
    wm[1] = weights[1]*midpoint_distances[1];

    w_prefix[1] = weights[0];
    wm_prefix[1] = wm[0];
    for (int i{2}; i < total_steps; i++) {
        wm[i] = weights[i]*midpoint_distances[i];
        w_prefix[i] = w_prefix[i-1] + weights[i-1];
        wm_prefix[i] = wm_prefix[i-1] + wm[i-1];
    }
    w_total  = w_prefix[total_steps-1] + weights[total_steps-1];
    wm_total = wm_prefix[total_steps-1] + wm[total_steps-1];
}

// adapted from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
void distloss_backward_pass(
        float* __restrict__ weights,
        float* __restrict__ midpoint_distances,
        float* __restrict__ intervals,
        float* __restrict__ wm,
        float* __restrict__ w_prefix,
        float* __restrict__ wm_prefix,
        float& __restrict__ w_total,
        float& __restrict__ wm_total,
        int& __restrict__ total_steps,
        //Output
        float* __restrict__ grad_arr) {
    
    for (int i{0}; i < total_steps; i++) {
        grad_arr[i] = ((1/3) * intervals[i] * 2 * weights[i]) //grad_uni
            + 2 * (midpoint_distances[i] * (2*w_prefix[i] - weights[i] - w_total) + (-2* wm_prefix[i]) + wm_total + wm[i]); //grad_bi
    }

}


// BEGIN KERNELS
__launch_bounds__(DISTLOSS_RAY_CUDA_THREADS, 0)
__global__ void distloss_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        PackedGridOutputGrads grads) {

    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE); //Global thread id? (E)
    const int ray_id = tid;                // Global ray id
    const int ray_blk_id = threadIdx.x;    // Local ray id, threadIdx.x is threadid local to block
    
    __shared__ SingleRaySpec ray_spec[DISTLOSS_RAY_CUDA_THREADS];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());

    __shared__ float ray_length[DISTLOSS_RAY_CUDA_THREADS];
    __shared__ int max_steps[DISTLOSS_RAY_CUDA_THREADS];
    __shared__ int total_steps[DISTLOSS_RAY_CUDA_THREADS];

    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id); // sets ray_spec tmin and tmax
    ray_length[ray_blk_id] = ray_spec[ray_blk_id].tmax - ray_spec[ray_blk_id].tmin;
    if (ray_length[ray_blk_id] < 0) return;

    max_steps[ray_blk_id] = ceil(ray_length[ray_blk_id]/opt.step_size)+1;

    float* weights = new float[max_steps[ray_blk_id]]{0};
    float* midpoint_distances = new float[max_steps[ray_blk_id]]{0};
    float* intervals = new float[max_steps[ray_blk_id]]{0};

    total_steps[ray_blk_id] = trace_ray_distloss( 
        grid,
        ray_spec[ray_blk_id],
        opt,
        ray_length[ray_blk_id],
        weights,
        midpoint_distances,
        intervals);

    // Forward pass 
    float* wm = new float[total_steps[ray_blk_id]]{0};

    float* w_prefix = new float[total_steps[ray_blk_id]]{0};  // w_cumsum rightshifted 1
    float* wm_prefix = new float[total_steps[ray_blk_id]]{0}; // wm_cumsum rightshifted 1

    __shared__ float w_total[DISTLOSS_RAY_CUDA_THREADS];
    __shared__ float wm_total[DISTLOSS_RAY_CUDA_THREADS];

    // Fill arrays
    distloss_forward_pass(
        weights, 
        midpoint_distances,
        wm,
        w_prefix,
        wm_prefix,
        w_total[ray_blk_id],
        wm_total[ray_blk_id],
        total_steps[ray_blk_id]);
    
    // Fill grad_arr with gradients for each sample point
    float* grad_arr = new float[total_steps[ray_blk_id]]{0};
    distloss_backward_pass(
        weights, 
        midpoint_distances,
        intervals,
        wm,
        w_prefix,
        wm_prefix,
        w_total[ray_blk_id],
        wm_total[ray_blk_id],
        total_steps[ray_blk_id],
        //output
        grad_arr);

    // Apply gradients from each sample point to gradient voxels
    trace_ray_backward_distloss(
        grid,
        grad_arr,
        ray_spec[ray_blk_id],
        opt,
        grads
    );

    delete[] weights;
    delete[] midpoint_distances;
    delete[] intervals;

    delete[] wm;
    delete[] w_prefix;
    delete[] wm_prefix;

    delete[] grad_arr;
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE); //Global thread id? (E)
    const int ray_id = tid >> 5;                //Actual global ray id
    const int ray_blk_id = threadIdx.x >> 5;    //Local ray id, threadIdx.x is threadid local to block
    const int lane_id = threadIdx.x & 0x1F;     //What is lane_id? (E) Spherical harmonic coefficient id

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,             // Calculates value of SH function in ray direction
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]    //Output
                 ); 
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id); // sets ray_spec tmin and tmax
    __syncwarp((1U << grid.sh_data_dim) - 1); // "synchronize threads in a warp and provide a memory fence."

    trace_ray_cuvol( //Finds color by raytracing for each SH coefficient.
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id, //SH coefficient
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        float* __restrict__ out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, cam.height * cam.width * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    const int ix = ray_id % cam.width;
    const int iy = ray_id / cam.width;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];

    cam2world_ray(ix, iy, cam, ray_spec[ray_blk_id].dir, ray_spec[ray_blk_id].origin);
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out + ray_id * 3,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output, //rgb_gt.data_ptr 
    const float* __restrict__ color_cache, //rgb_out.data_ptr 
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb, //TRUE
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr, 
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };
    if (lane_id < grid.basis_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id, //SH function value in direction
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]); 
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id); // set ray tmin, tmax
    }

    //Calculates residuals
    float grad_out[3]; 
    if (grad_out_is_rgb) { // TRUE
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) { //NCHW: data_pos = n * CHW + c * HW + h * W + w
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i]; //out - gt
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1); // Sync threads
    trace_ray_cuvol_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
    calc_sphfunc_backward(
                 grid, lane_id,
                 ray_id,
                 vdir,
                 sphfunc_val[ray_blk_id],
                 grad_sphfunc_val[ray_blk_id],
                 grads.grad_basis_out);
}

__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out[ray_id].data());
}

__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, cam.height * cam.width);
    if (log_transmit[ray_id] < -25.f) return;
    const int ix = ray_id % cam.width;
    const int iy = ray_id / cam.width;
    SingleRaySpec ray_spec;
    cam2world_ray(ix, iy, cam, ray_spec.dir, ray_spec.origin);
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out + ray_id * 3);
}

__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_backward_kernel(
        PackedSparseGridSpec grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        const float* __restrict__ accum,
        bool grad_out_is_rgb,
        float sparsity_loss,
        // Outputs
        PackedGridOutputGrads grads) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    render_background_backward(
        grid,
        grad_out,
        ray_spec,
        opt,
        log_transmit[ray_id],
        accum[ray_id],
        sparsity_loss,
        grads);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_expected_term_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float* __restrict__ out) {
        // const PackedSparseGridSpec& __restrict__ grid,
        // SingleRaySpec& __restrict__ ray,
        // const RenderOptions& __restrict__ opt,
        // float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_expected_term(
        grid,
        ray_spec,
        opt,
        out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_sigma_thresh_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float sigma_thresh,
        float* __restrict__ out) {
        // const PackedSparseGridSpec& __restrict__ grid,
        // SingleRaySpec& __restrict__ ray,
        // const RenderOptions& __restrict__ opt,
        // float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_sigma_thresh(
        grid,
        ray_spec,
        opt,
        sigma_thresh,
        out + ray_id);
}

}  // namespace device

torch::Tensor _get_empty_1d(const torch::Tensor& origins) {
    auto options =
        torch::TensorOptions()
        .dtype(origins.dtype())
        .layout(torch::kStrided)
        .device(origins.device())
        .requires_grad(false);
    return torch::empty({origins.size(0)}, options);
}

}  // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();


    const auto Q = rays.origins.size(0);

    torch::Tensor results = torch::empty_like(rays.origins);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit;
    if (use_background) {
        log_transmit = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
                grid, rays, opt,
                // Output
                results.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        // printf("RENDER BG\n");
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                results.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
    return results;
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();


    const auto Q = cam.height * cam.width;
    auto options =
        torch::TensorOptions()
        .dtype(grid.sh_data.dtype())
        .layout(torch::kStrided)
        .device(grid.sh_data.device())
        .requires_grad(false);

    torch::Tensor results = torch::empty({cam.height, cam.width, 3}, options);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit;
    if (use_background) {
        log_transmit = torch::empty({cam.height, cam.width}, options);
    }

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_image_kernel<<<blocks, cuda_n_threads>>>(
                grid,
                cam,
                opt,
                // Output
                results.data_ptr<float>(),
                use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        // printf("RENDER BG\n");
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_image_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                cam,
                opt,
                log_transmit.data_ptr<float>(),
                results.data_ptr<float>());
    }

    CUDA_CHECK_ERRORS;
    return results;
}

void volume_render_cuvol_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit, accum;
    if (use_background) {
        log_transmit = _get_empty_1d(rays.origins);
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);
        device::render_ray_backward_kernel<<<blocks,
            cuda_n_threads_render_backward>>>(
                    grid,
                    grad_out.data_ptr<float>(),
                    color_cache.data_ptr<float>(),
                    rays, opt,
                    false,
                    nullptr,
                    0.f,
                    0.f,
                    // Output
                    grads,
                    use_background ? accum.data_ptr<float>() : nullptr,
                    use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                grad_out.data_ptr<float>(),
                color_cache.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                false,
                0.f,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float beta_loss,
        float sparsity_loss,
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0; //true?(E)
    bool need_log_transmit = use_background || beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit) {
        log_transmit = _get_empty_1d(rays.origins);
    }
    if (use_background) {
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                //Output? (E)
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                // Output
                grads,
                use_background ? accum.data_ptr<float>() : nullptr,
                nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                true,
                sparsity_loss,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused_distloss(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float beta_loss,
        float sparsity_loss,
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0; //true?(E)
    bool need_log_transmit = use_background || beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit) {
        log_transmit = _get_empty_1d(rays.origins);
    }
    if (use_background) {
        accum = _get_empty_1d(rays.origins);
    }

    //Create distortion loss quadratic form matrix and weight vector
    //torch::Tensor** p_quad_mat_arr = new torch::Tensor*[rays.origins.size(0)]; //CANT BE THIS BIG!!!
    //torch::Tensor** p_weights_arr = new torch::Tensor*[rays.origins.size(0)];

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>( //Each block does 4 rays of 32 threads
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                //Output? (E)
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    //Add distortion loss kernel here!
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, DISTLOSS_RAY_CUDA_THREADS);
        device::distloss_kernel<<<blocks, DISTLOSS_RAY_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                //Output
                grads);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                // Output
                grads,
                use_background ? accum.data_ptr<float>() : nullptr,
                nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                true,
                sparsity_loss,
                // Output
                grads);
    }
    printf("volume_render_cuvol_fused_distloss finished");
    CUDA_CHECK_ERRORS;
}

void distloss_grad(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        float scaling,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);
    
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, DISTLOSS_RAY_CUDA_THREADS);
    device::distloss_kernel<<<blocks, DISTLOSS_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            //Output
            grads);
            
    CUDA_CHECK_ERRORS;
}

torch::Tensor volume_render_expected_term(SparseGridSpec& grid,
        RaysSpec& rays, RenderOptions& opt) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_expected_term_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            results.data_ptr<float>()
        );
    return results;
}

torch::Tensor volume_render_sigma_thresh(SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        float sigma_thresh) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_sigma_thresh_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            sigma_thresh,
            results.data_ptr<float>()
        );
    return results;
}
