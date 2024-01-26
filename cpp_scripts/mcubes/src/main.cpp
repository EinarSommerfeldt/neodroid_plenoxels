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

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
    Eigen::MatrixXd K = readMatrix("C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/K.txt");
    std::string path = "C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/pose";
    std::vector<Eigen::Matrix<double, 4, 4>> Tvec = read_Tvec(path);

    marching_cubes(K, Tvec, )
}