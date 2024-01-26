#pragma once

#include <Eigen/Dense>
#include <math.h>

static Eigen::MatrixXd base_cube_vertices{
    {0, -1, 0, 0, -1, -1, 0, -1},
    {0, 0, -1, 0, -1, 0, -1, -1},
    {0, 0, 0, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1}
};

/*
Colmap coordinate convention
*/
class Cuboid {
public:
    double x;
    double y;
    double z;
    double width;
    double height; 
    double depth;
    
    Cuboid(double x, double y, double z, double width, double height, double depth)
    :x{x}, y{y}, z{z}, width{width}, height{height}, depth{depth} {

    };

    Eigen::MatrixXd Cuboid::to_vertices() {

        Eigen::MatrixXd cube_vertices;
        cube_vertices = base_cube_vertices.array().rowwise() 
            * Eigen::RowVector4d{width, height, depth,1}.array();

        Eigen::Vector3d T{x,y,z};
        Eigen::Matrix4d Trans;
        Trans.setIdentity();
        Trans.block<3,1>(0,3) = T;

        return Trans*cube_vertices;
    };

};