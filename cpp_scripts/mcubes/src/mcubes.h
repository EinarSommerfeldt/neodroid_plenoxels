#pragma once

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

#include <vector>

#include "cuboid.h"

Eigen::Vector2d project(Eigen::Matrix<double, 3, 3> K, Eigen::MatrixXd X);

void marching_cubes(
    Eigen::Matrix<double, 3, 3> K,
    std::vector<Eigen::Matrix<double, 4, 4>> T_vec,
    Cuboid cube,
    double step_length
);