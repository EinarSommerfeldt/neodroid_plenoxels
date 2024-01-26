#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include <math.h>

#include "mcubes.h"
#include "cuboid.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
    Cuboid c{0,0,0,1,1,1};
    std::cout << c.to_vertices();
}