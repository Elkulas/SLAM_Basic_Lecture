//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

#include <sophus/so3.h>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    Eigen::JacobiSVD<Matrix3d> svd(E, ComputeFullU | ComputeFullV);
    Eigen::Vector3d vsigma = svd.singularValues();
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "sigma = " << vsigma << endl;
    // 将sigma从vector形式转化成矩阵形式
    Eigen::Matrix3d msigma;
    msigma << (vsigma(0,0) + vsigma(1,0))/2, 0, 0,
               0, (vsigma(1,0) + vsigma(0,0))/2, 0,
               0 , 0, 0 ;
    cout << "Matrix sigma = " << endl << msigma << endl << "OKKKKKK" << endl;
    
    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    // Z轴顺时针90度旋转矩阵
    Matrix3d Rz_c = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    // Z轴逆时针90度旋转矩阵
    Matrix3d Rz_a = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    cout << "Rotation Matrix clockwise = " << endl << Rz_c <<endl;
    cout << "Rotation Matrix anticlockwise = " << endl << Rz_a <<endl;

    Matrix3d t_wedge1;
    Matrix3d t_wedge2;
    t_wedge1 = U * Rz_c * msigma * U.transpose();
    t_wedge2 = U * Rz_a * msigma * U.transpose();

    Matrix3d R1;
    Matrix3d R2;

    R1 = U * Rz_c.transpose() * V.transpose();
    R2 = U * Rz_a.transpose() * V.transpose();

    // END YOUR CODE HERE

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}