//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../src/p3d.txt";
string p2d_file = "../src/p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    fstream i3dfile(p3d_file);
    fstream i2dfile(p2d_file);
    string line;
    while(getline(i3dfile,line))
    {
        stringstream record(line);
        Vector3d vec3d;
        
        for(int i = 0; i<3 ; i++)
            record >> vec3d[i];
        p3d.push_back(vec3d);
        
    }
    while(getline(i2dfile,line))
    {
        stringstream record(line);
        Vector2d vec2d;
        for(int i = 0; i<2 ; i++)
            record >> vec2d[i];
        p2d.push_back(vec2d);
    }

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();

    std::cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
        // START YOUR CODE HERE 
        Vector3d p3 = p3d[i];
        Vector2d p2 = p2d[i];

        Vector3d P = T_esti * p3;
        // P = [x',y',z']
        double x = P[0];
        double y = P[1];
        double z = P[2];

        Vector2d p2_;
        // 投影过后的u，v
        p2_(0,0) = fx * x / z + cx;
        p2_(1,0) = fy * y / z + cy;
        // cout << "投影过后的uv = " << endl << p2_ << endl;
        Vector2d e = p2 - p2_;
        // 所有cost累加
        cost += (e[0] * e[0] + e[1] * e[1]);

	    // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE 
        // 根据雅各比矩阵推导得到的模式进行书写
        J << -fx/z, 0 , fx * x/pow(z,2), fx*x*y/pow(z,2), -fx-fx*x*x/pow(z,2), fx*y/z,
             0, -fy/z, fy*y/pow(z,2), fy+fy*pow(y,2)/pow(z,2), -fy*x*y/pow(z,2), -fy*x/z;
        // cout << "雅各比矩阵输出 = " << endl << J << endl;

	    // END YOUR CODE HERE

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b);

        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3::exp(dx) * T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
