#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "../image/left.png";
string disparity_file = "../image/disparity.png";
boost::format fmt_others("../image/%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);
    if(left_img.size || disparity_img.size)
        cout << "IMport images OK!" << endl;

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000;
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3 T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);    // first you need to test single layer
        // DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
) {

    // parameters
    // 窗口大小 8x8
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++) {

            // compute the projection in the second image
            // TODO START YOUR CODE HERE
            // 定义投影过后的像素点坐标uv
            float u =0, v = 0;
            // 1. 将参考坐标系的点通过相机投影模型还原成三维点坐标
            // 判断参考平面选点是否超出图像范围
            float u_ref = px_ref[i].x();
            float v_ref = px_ref[i].y();
            if (u_ref - half_patch_size < 0 || u_ref + half_patch_size >= img1.cols || 
            v_ref - half_patch_size < 0 || v_ref + half_patch_size >= img1.rows ) continue;

            // 将像素平面点转化到实际像平面上以及三维坐标点
            double z_ref = depth_ref[i];
            double x_ref = (u_ref - cx) * z_ref/fx;
            double y_ref = (v_ref - cy) * z_ref/fy;

            // 生成三维点坐标
            Eigen::Vector3d pc1(x_ref, y_ref, z_ref);

            // 通过位姿转换为当前坐标下的三维点
            Eigen::Vector3d pc2 = T21 * pc1;

            // 投影矩阵计算当前图像的投影位置
            u = float(pc2.x() * fx / pc2.z() + cx);
            v = float(pc2.y() * fy / pc2.z() + cy);

            // 判断获得的投影坐标是否在当前图像上
            if(u - half_patch_size < 0 || u + half_patch_size >= img2.cols || 
            v - half_patch_size < 0 || v + half_patch_size >= img2.rows ) continue;

            nGood++;
            
            goodProjection.push_back(Eigen::Vector2d(u, v));

            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error =GetPixelValue(img1,u_ref + x,v_ref + y) - GetPixelValue(img2,u + x,v + y);

                    Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra
                    Eigen::Vector2d J_img_pixel;    // image gradients

                    // 计算图像梯度雅克比J_img_pixel, 2x1, 需要转置成为最后的雅克比1x2
                    // 判断所取的像素位置是否在图像范围内（可以不用判断）
                    float u2 = float(u + x);
                    float v2 = float(v + y);
                    float z2 = pc2.z();
                    if (u2 - 1 < 0 || u2 + 1 > img2.cols || v2 -1 < 0 || v2 + 1 > img2.rows) continue;
                    J_img_pixel.x() = double(GetPixelValue(img2, u2+1, v2) - GetPixelValue(img2, u2-1, v2))/2;
                    J_img_pixel.y() = double(GetPixelValue(img2, u2, v2+1) - GetPixelValue(img2, u2, v2-1))/2;
                    // cout << "J_img_pixeal :::: " << J_img_pixel << endl;

                    // 计算重投影雅克比 2x6
                    //J << -fx/z, 0 , fx * x/pow(z,2), fx*x*y/pow(z,2), -fx-fx*x*x/pow(z,2), fx*y/z, 0, -fy/z, fy*y/pow(z,2), fy+fy*pow(y,2)/pow(z,2), -fy*x*y/pow(z,2), -fy*x/z;
                    J_pixel_xi(0,0) = - fx / pc2.z();
                    J_pixel_xi(0,1) = 0;
                    J_pixel_xi(0,2) = fx * pc2.x() / pow(pc2.z(), 2);
                    J_pixel_xi(0,3) = fx * pc2.x() * pc2.y() / pow(pc2.z(), 2);
                    J_pixel_xi(0,4) = -fx - fx * pow(pc2.x(), 2) / pow(pc2.z(), 2);
                    J_pixel_xi(0,5) = fx * pc2.y() / pc2.z();
                    J_pixel_xi(1,0) = 0;
                    J_pixel_xi(1,1) = - fy / pc2.z();
                    J_pixel_xi(1,2) = fy * pc2.y() / pow(pc2.z(), 2);
                    J_pixel_xi(1,3) = fy + fy*pow(pc2.y(), 2)/pow(pc2.z(), 2);
                    J_pixel_xi(1,4) = - fy * pc2.x() * pc2.y() / pow(pc2.z(), 2);
                    J_pixel_xi(1,5) = - fy * pc2.x() / pc2.z();

                    // total jacobian
                    // 计算总的雅克比 6x1 = 6x2 * 2x1
                    Vector6d J = J_pixel_xi.transpose() * J_img_pixel;

                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }
            // END YOUR CODE HERE
        }

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update;
        update = H.ldlt().solve(b);
        cout << "UPdate:::::::::::" << update << endl;
        T21 = Sophus::SE3::exp(update) * T21;
        // END YOUR CODE HERE

        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (auto &px: goodProjection) {
        cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    cv::imshow("reference", img1_show);
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE

    // END YOUR CODE HERE

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // TODO START YOUR CODE HERE
        // scale fx, fy, cx, cy in different pyramid levels

        // END YOUR CODE HERE
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
