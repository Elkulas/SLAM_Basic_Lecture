#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "../image/1.png";  // first image
string file_2 = "../image/2.png";  // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
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

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, true);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    // TODO START YOUR CODE HERE (~8 lines)
                    // 开始对8x8窗口内像素进行遍历工作
                    // 获得template上的坐标
                    float u1 = float(kp.pt.x + x);
                    float v1 = float(kp.pt.y + y);
                    // 获得第二幅图像上更新的坐标
                    float u2 = float(u1 + dx);
                    float v2 = float(v1 + dy);

                    double error = 0;
                    Eigen::Vector2d J;  // Jacobian
                    if (inverse == false) {
                        // Forward Jacobian 这边就是计算像素梯度的雅克比
                        // dIx/du 计算第二幅图像中在update之后的点x轴上的图像梯度
                        J.x() = double(GetPixelValue(img2, u2+1, v2) - GetPixelValue(img2, u2-1, v2))/2;
                        // 计算第二幅图像中在update之后的点y轴上的图像梯度
                        J.y() = double(GetPixelValue(img2, u2, v2+1) - GetPixelValue(img2, u2, v2-1))/2;
                        // 计算error
                        error = double(GetPixelValue(img2, u2, v2) - GetPixelValue(img1, u1, v1));

                    } else {
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J.x() = double(GetPixelValue(img1, u1 + 1, v1) - GetPixelValue(img1, u1 - 1, v1))/2;
                        J.y() = double(GetPixelValue(img1, u1, v1 + 1) - GetPixelValue(img1, u1, v1 - 1))/2;
                        error = double(GetPixelValue(img2, u2, v2) - GetPixelValue(img1, u1, v1));
                    }

                    // compute H, b and set cost;
                    // 这里注意之前求解得到的J是2x1的，但实际上雅克比矩阵应该是1x2的
                    //cout << "之前:::::: "<<endl << J <<endl;
                    Eigen::Matrix<double, 1, 2> J12;
                    J12 = J.transpose();
                    //cout << "之后：：：：：：" << endl << J12 << endl;
                    // H = J^T*J
                    H += J12.transpose() * J12;
                    // b = - J^T * error
                    b += -J12.transpose() * error;
                    cost += pow(error, 2);
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;
            update = H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    cout << "= = = = GENERATING PYRAMIDS = = = = " << endl;
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++) {
        // 生成不同分辨率的图像，并存放到pyr容器中
        Mat img1_temp, img2_temp;
        // 使用resize函数对图像进行分辨率重构
        cv::resize(img1, img1_temp, cv::Size(img1.cols * scales[i], img1.rows * scales[i]));
        cv::resize(img2, img2_temp, cv::Size(img2.cols * scales[i], img2.rows * scales[i]));
        pyr1.push_back(img1_temp);
        pyr2.push_back(img2_temp);
        // 输出进行检查
        cout << "Pyramid " << i << "has " << img1_temp.cols << " x " << img2_temp.rows << endl;

    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    // 本质上说就是从较低分辨率图像开始检测光流，然后传递到下一层
    // 从金字塔高层到底层
    vector<KeyPoint> kp2_now;
    vector<KeyPoint> kp2_last;
    vector<bool> vsucc;
    for (int i = pyramids -1; i >= 0; i--)
    {
        cout << "At pyramid " << i << endl;
        // 生成一个对template图像特征点进行缩放过后的容器
        vector<KeyPoint> kp1_now;
        // 对所有kp1进行尺度上的转化
        
        for(int j = 0; j < kp1.size(); j++)
        {
            KeyPoint kp1_temp;
            kp1_temp = kp1[j]; // 这里之前写了一个bug，是kp1[j],不是kp1[i]
            kp1_temp.pt = kp1_temp.pt * scales[i];
            kp1_now.push_back(kp1_temp);
            // 对所有上一层的kp2进行尺度上的转化, 进行放大
            if (i < pyramids -1)
            {
                KeyPoint kp2_temp;
                kp2_temp = kp2_last[j];
                kp2_temp.pt /= pyramid_scale;
                kp2_now.push_back(kp2_temp);
            }
        }
        vsucc.clear();
        // 获得该层第二幅图像中的对应特征点
        OpticalFlowSingleLevel(pyr1[i], pyr2[i], kp1_now, kp2_now, vsucc, false);
        cout<<"pyramid: "<<i<<" kp2_last size: "<<kp2_last.size()<<"kp2_nowsize "<<kp2_now.size()<<endl;
        if(i == 3)
        for(int k = 0; k < kp2_now.size(); k++) 
        cout << kp2_now[k].pt << endl;
        // 将上一层的kp2存入last，将now清空
        kp2_last.clear();
        kp2_last.swap(kp2_now);
    }
    kp2 = kp2_last;
    success = vsucc;

    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}
