#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <filesystem>

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
