#include "mcubes.h"

#include <iostream>
#include <string>

#include "math.h"

Eigen::Matrix2Xd project(Eigen::Matrix<double, 3, 3> K, Eigen::MatrixXd X) {

    Eigen::Matrix3Xd  uvw = K*X({0,1,2},Eigen::all);
    return uvw.colwise().hnormalized();
}

Cuboid find_bbox(std::vector<Eigen::Matrix<double, 4, 4>> T_vec){
    Eigen::Matrix3Xd camera_positions(3,T_vec.size());

    Eigen::Vector3d cam_pos;
    int i{0};
    for (Eigen::Matrix<double, 4, 4> T : T_vec) {
        cam_pos = T({0,1,2},3);
        camera_positions.col(i) = cam_pos;
        i++;
    }
    Eigen::Vector<double, 6> maxmin_values;
    maxmin_values({0,1,2}) = camera_positions.rowwise().maxCoeff();
    maxmin_values({3,4,5}) = camera_positions.rowwise().minCoeff();
    
    double side_length = std::ceil(2*maxmin_values.array().abs().maxCoeff());
    
    Eigen::Vector3d center = (maxmin_values({0,1,2})-maxmin_values({3,4,5})).array().abs();
    center = center/2 + maxmin_values({3,4,5});

    double s = 0.1;
    Cuboid cube{center(0)-side_length/2,center(1)-side_length/2,center(2)-side_length/2,side_length,side_length,side_length};
    return cube;
}

Eigen::Tensor<uint8_t,3> marching_cubes(
    Eigen::Matrix<double, 3, 3> K,
    std::vector<Eigen::Matrix<double, 4, 4>> T_vec,
    Cuboid bbox,
    double step_length,
    int img_width,
    int img_height
) {
    int w_its = static_cast<int>(std::ceil(bbox.width/step_length));
    int h_its = static_cast<int>(std::ceil(bbox.height/step_length));
    int d_its = static_cast<int>(std::ceil(bbox.depth/step_length));

    std::cout << "bbox cubes: " << w_its <<" " << h_its <<" "<< d_its <<"\n";

    Eigen::Tensor<uint8_t, 3> mask(w_its, h_its, d_its);
    mask.setZero();

    Cuboid cube = bbox;
    cube.width = step_length;
    cube.height = step_length;
    cube.depth = step_length;

    Eigen::MatrixXd cube_world;
    Eigen::MatrixXd cube_cam;
    //Marches cubes through bounding box. A cube has to be entirely visible to a camera to be counted.
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
                    if (!(cube_cam.row(2).array() >= 0).isOnes()) continue; //Check Z value in front of cam for entire cube
                    Eigen::Matrix2Xd points = project(K,cube_cam); 
                    auto u_check = (points.row(0).array()>0 && points.row(0).array()<img_width);
                    auto v_check = (points.row(1).array()>0 && points.row(1).array()<img_height);
                    if (u_check.isOnes() && v_check.isOnes()) { //All cube points inside image.
                        mask(w,h,d)++;}
                    /*
                    for (auto col : points.colwise()) {
                        if (col(0) > 0 && col(0) < img_width && col(1) > 0 && col(1) < img_height) {
                            mask(w,h,d)++;
                            break;
                        }
                    }*/
                    if (mask(w,h,d) > 50) {
                        break; 
                    }
                }
            }
        }
    }
    std::cout << mask;
    return mask;
}