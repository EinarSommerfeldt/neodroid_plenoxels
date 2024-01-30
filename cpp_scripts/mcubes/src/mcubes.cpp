#include "mcubes.h"
#include "math.h"
#include <iostream>


Eigen::Matrix2Xd project(Eigen::Matrix<double, 3, 3> K, Eigen::MatrixXd X) {

    Eigen::Matrix3Xd  uvw = K*X({0,1,2},Eigen::all);
    return uvw.colwise().hnormalized();
}

Eigen::Tensor<uint8_t,3> marching_cubes(
    Eigen::Matrix<double, 3, 3> K,
    std::vector<Eigen::Matrix<double, 4, 4>> T_vec,
    Cuboid volume,
    double step_length,
    int img_width,
    int img_height
) {
    int w_its = static_cast<int>(std::ceil(volume.width/step_length));
    int h_its = static_cast<int>(std::ceil(volume.height/step_length));
    int d_its = static_cast<int>(std::ceil(volume.depth/step_length));

    Eigen::Tensor<uint8_t, 3> mask(w_its, h_its, d_its);
    mask.setZero();

    Cuboid cube = volume;
    cube.width = step_length;
    cube.height = step_length;
    cube.depth = step_length;

    Eigen::MatrixXd cube_world;
    Eigen::MatrixXd cube_cam;
    for(int w{0}; w < w_its; w++) {
        std::cout << w << "\n";
        cube.x = cube.x + w*step_length;
        for(int h{0}; h < h_its; h++) {
            cube.y = cube.y + h*step_length;
            for(int d{0}; d < d_its; d++) {
                cube.z = cube.z + d*step_length;
                cube_world = cube.to_vertices();
                for(Eigen::Matrix<double, 4, 4> T : T_vec) {
                    cube_cam = T*cube_world;
                    Eigen::Matrix2Xd points = project(K,cube_cam); 
                    for (auto col : points.colwise()) {
                        if (col(0) > 0 && col(0) < img_width && col(1) > 0 && col(1) < img_height) {
                            mask(w,h,d)++;
                            break;
                        }
                    }
                    if (mask(w,h,d) > 4) break;
                }
            }
        }
    }
    std::cout << mask;
    return mask;
}