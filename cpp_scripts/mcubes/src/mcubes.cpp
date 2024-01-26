#include "mcubes.h"
#include "math.h"

Eigen::Vector2d project(Eigen::Matrix<double, 3, 3> K, Eigen::MatrixXd X) {
    Eigen::VectorXd  uvw = K*X(Eigen::seq(0,2),Eigen::all);
    uvw = uvw.array().rowwise() / uvw(Eigen::last, Eigen::all).array();
    
    return uvw({0,1}, Eigen::all);
}

void marching_cubes(
    Eigen::Matrix<double, 3, 3> K,
    std::vector<Eigen::Matrix<double, 4, 4>> T_vec,
    Cuboid cube,
    double step_length
) {
    int iterations = std::round(cube.width/step_length);
    uint8_t* p_mask = new uint8_t[iterations/sizeof(uint8_t)];
    delete[] p_mask;
}