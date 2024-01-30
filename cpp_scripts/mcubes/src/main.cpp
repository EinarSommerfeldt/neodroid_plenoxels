#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include <math.h>



#include "mcubes.h"
#include "cuboid.h"
#include "utils.h"


int main()
{
    Eigen::MatrixXd K = readMatrix("C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/K.txt");
    std::string path = "C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/pose";
    std::vector<Eigen::Matrix<double, 4, 4>> Tvec = read_Tvec(path);

    double s = 0.1;
    Cuboid cube{0.2,0.3,1.2,s*3,s,s*2};
    marching_cubes(K, Tvec, cube,0.01, 1008, 756);
}