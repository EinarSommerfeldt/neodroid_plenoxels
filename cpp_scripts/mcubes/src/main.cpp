#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include <math.h>
#include <time.h>
#include <sstream>

#include "mcubes.h"
#include "cuboid.h"
#include "utils.h"


int main()
{
    Eigen::MatrixXd K = readMatrix("C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/K.txt");
    std::string path = "C:/Users/einarjso/neodroid_plenoxels/python_scripts/roi/data/pose";
    std::vector<Eigen::Matrix<double, 4, 4>> Tvec = read_Tvec(path);

    double voxel_size = 0.5;
    Cuboid bbox = find_bbox(Tvec);
    Eigen::Tensor<uint8_t, 3> mask = marching_cubes(K, Tvec, bbox, voxel_size, 1008, 756);
    

    time_t seconds;
    seconds = time (NULL);
    std::string tensorpath = "C:/Users/einarjso/neodroid_plenoxels/cpp_scripts/mcubes/data/tensor" + std::to_string(seconds) + ".txt";
    write3DTensorToFile(tensorpath, mask);
    std::string bboxpath = "C:/Users/einarjso/neodroid_plenoxels/cpp_scripts/mcubes/data/bbox" + std::to_string(seconds) + ".txt";
    writeBboxToFile(bboxpath, bbox, voxel_size);
}