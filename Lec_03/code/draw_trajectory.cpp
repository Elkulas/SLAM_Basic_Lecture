#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file
string trajectory_file = "../trajectory.txt";
string estimate_file = "../estimated.txt";
string ground_file = "../groundtruth.txt";

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_es;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_gt;

    /// implement pose reading code
    // start your code here (5~10 lines)

    ifstream infile(estimate_file);

    double t, tx, ty, tz, qx, qy, qz, qw;
    
    string line;

    if (infile)
    {
        while(getline(infile, line))
        {
            stringstream record(line);
            record >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw ;
            Eigen::Vector3d t(tx, ty, tz);
            Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();

            Sophus::SE3 SE3_t(q,t);
            poses_es.push_back(SE3_t);
        }
    }

    infile.close();

    ifstream infile2(ground_file);

    //double t, tx, ty, tz, qx, qy, qz, qw;
    
    //string line;

    if (infile2)
    {
        while(getline(infile2, line))
        {
            stringstream record(line);
            record >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw ;
            Eigen::Vector3d t(tx, ty, tz);
            Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();

            Sophus::SE3 SE3_t(q,t);
            poses_gt.push_back(SE3_t);
        }
    }

    infile2.close();

    double RMSE = 0;
    Eigen::Matrix<double, 6, 1> se3;
    vector<double> err;

    for (int i = 0; i < poses_es.size(); i++)
    {
        se3 = (poses_gt[i].inverse() * poses_es[i]).log();

        cout << "se3: " << i << " is " << se3.transpose() << endl;

        err.push_back(se3.squaredNorm());
    }

    for (int i = 0; i < err.size(); i ++ )
    {
        RMSE = RMSE + err[i];
    }

    RMSE = sqrt(RMSE/err.size());

    cout << "RES::::::" << RMSE << endl;

    


    // end your code here

    // draw trajectory in pangolin
    DrawTrajectory(poses_es, poses_gt);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2) {
    if (poses.empty() || poses2.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        glLineWidth(2);
        for (size_t i = 0; i < poses2.size() - 1; i++) {
            glColor3f(1 - (float) i / poses2.size(), 0.0f, (float) i / poses2.size());
            glBegin(GL_LINES);
            auto p1 = poses2[i], p2 = poses2[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}