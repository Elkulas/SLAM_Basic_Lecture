//
//

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "/home/jjj/NGCLAB/SensorDriver/udistort/image/test2.bmp";   // 请确保路径正确

int main(int argc, char **argv) {

    // 本程序需要你自己实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
    // 畸变参数
    // double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // // 内参
    // double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    double k1 = -0.482575219159269, k2 = 0.305372657009255, k3 = -0.116353890768303, p1 = 2.875811241413620e-04, p2 = -9.783970247389863e-05;
    // 内参
    double fx = 4.401844147106322e+02, fy = 4.398887665018974e+02, cx = 3.246975373514975e+02, cy = 2.543902325439119e+02;


    cv::Mat image = cv::imread(image_file,0);   // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {

            double u_distorted = 0, v_distorted = 0;
            // TODO: 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted) (~6 lines)
            // start your code here
            
            // 使用内参将像素平面转化到实际成像平面上的点
            double x = (u - cx)/fx;
            double y = (v - cy)/fy;

            // 计算到光心的距离r
            double r = sqrt(pow(x,2) + pow(y,2));

            // 计算矫正之后的图像坐标
            double x_distorted = x * (1 + k1 * pow(r,2) + k2*pow(r,4) + k3*pow(r,6)) + 2*p1*x*y + p2*(pow(r,2)+2*pow(x,2));
            double y_distorted = y * (1 + k1 * pow(r,2) + k2*pow(r,4) + k3*pow(r,6)) + p1*(pow(r,2)+2*pow(y,2)) + 2*p2*x*y;

            // 将成像平面点转化为像素平面的点
            u_distorted = x_distorted * fx + cx;
            v_distorted = y_distorted * fy + cy;


            // end your code here

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }

    // 画图去畸变后图像
    cv::imshow("image undistorted", image_undistort);
    cv::waitKey();

    return 0;
}
