#pragma once

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

#include <vector>

#include "cuboid.h"

Eigen::Matrix2Xd project(Eigen::Matrix<double, 3, 3> K, Eigen::MatrixXd X);

Cuboid find_bbox(std::vector<Eigen::Matrix<double, 4, 4>> T_vec);

Eigen::Tensor<uint8_t,3> marching_cubes(
    Eigen::Matrix<double, 3, 3> K,
    std::vector<Eigen::Matrix<double, 4, 4>> T_vec,
    Cuboid volume,
    double step_length,
    int img_width,
    int img_height
);