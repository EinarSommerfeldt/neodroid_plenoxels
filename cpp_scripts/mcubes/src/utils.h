#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <filesystem>

#include "cuboid.h"

using namespace std;
namespace fs = std::filesystem;



Eigen::MatrixXd readMatrix(std::string filename){
    int cols = 0, rows = 0;
    double buff[((int) 1e3)];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
};

std::vector<Eigen::Matrix<double, 4, 4>> read_Tvec(std::string folderpath) {
    std::vector<Eigen::Matrix<double, 4, 4>> Tvec;
    for (const auto & entry : fs::directory_iterator(folderpath)) {
        Tvec.push_back(readMatrix(entry.path().string()));
    }
        
    return Tvec;
}

template <typename T, int h, int w>
void writeMatrixToFile(std::string filepath, Eigen::Matrix<T,h,w> matrix) {
    std::ofstream file(filepath);
    if (file.is_open())
    {
        file << matrix;
    } else {
        std::cout << "Failed to write Matrix to filepath " << filepath << "\n";
    }
}

template <typename T>
void write3DTensorToFile(std::string filepath, Eigen::Tensor<T,3> tensor) {
    std::ofstream file(filepath);
    if (file.is_open())
    {
        file << tensor;
    } else {
        std::cout << "Failed to write Tensor to filepath " << filepath << "\n";
    }
}

void writeBboxToFile(std::string filepath, Cuboid cuboid, double step_size) {
    std::ofstream file(filepath);
    if (file.is_open())
    {
        file << cuboid.x << " " << cuboid.y << " " << cuboid.z << " " 
        << cuboid.x + cuboid.width << " "
        << cuboid.y + cuboid.height << " "
        << cuboid.z + cuboid.depth << " " 
        << step_size;
    } else {
        std::cout << "Failed to open  " << filepath << "\n";
    }
}