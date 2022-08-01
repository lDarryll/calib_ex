#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include <opencv2\imgproc\types_c.h> 1.091680007048003
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
using namespace cv;
using namespace std;

void drawPtsOnImage(vector<cv::Point2f> const &imgpts, cv::Mat &image_undistort)
{

    cv::Mat img = image_undistort;

    for (size_t i = 0; i < imgpts.size(); ++i)
    {
        // int cx = int(imgpts[i].x + 0.5);
        // int cy = int(imgpts[i].y + 0.5);
        int cx = int(imgpts[i].x);
        int cy = int(imgpts[i].y);

        cout << cx << "," << cy << endl;
        cv::circle(img, cv::Point(cx, cy), 2, cv::Scalar(0, 0, 255), -1);
        // cv::putText(img, to_string(i + 1), cv::Point(cx, cy), cv::FONT_HERSHEY_SIMPLEX,
        //             0.5, cv::Scalar(255, 0, 0), 2);
    }
    // cv::circle(img, cv::Point(607, 371), 2, cv::Scalar(0, 0, 255), -1);
    string src = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/temp_and_result/";
    imwrite(src + "/undis_1.jpg", img);
}

template <typename T>
void printVector(vector<T> &v)
{

    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}
//计算以板子中心为原点的各个点的坐标
vector<cv::Point2f> get_world_point_base_board()
{
    vector<cv::Point2f> points2f;
    vector<double> pitches; //板子相邻点之间的距离（左右点距离，上下点距离）
    pitches.assign(2, 2);
    vector<int> board_size; // 板子的大小
    board_size.assign(2, 7);
    // 以板子中心为原点计算出左上角第一个点的坐标
    float xs = -0.5 * (board_size[1] - 1) * pitches[1];
    float ys = -0.5 * (board_size[0] - 1) * pitches[0];
    for (int yi = 0; yi < board_size[0]; ++yi, ys += pitches[0])
    {
        float xss = xs;
        for (int xi = 0; xi < board_size[1]; ++xi, xss += pitches[1])
        {
            points2f.emplace_back(cv::Point2f{xss, ys});
        }
    }
    return points2f;
}
bool loadMatDist(cv::FileStorage &fs, cv::Mat &mtx, cv::Mat &dist)
{
    if (fs["mtx"].isNone() || fs["dist"].isNone())
    {
        cout << "Nodes mtx and/or dist are not in the camera file" << endl;
        return false;
    }
    fs["mtx"] >> mtx;
    fs["dist"] >> dist;

    return true;
}
// 加载相机内参
void loadIntrinsics(string const &filename, cv::Mat &mtx, cv::Mat &dist)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open camera file: " << filename << endl;
    }

    cout << "Read intrinsic from file: " << filename << endl;
    if (!loadMatDist(fs, mtx, dist))
    {
        cout << "Failed to load camera intrinsics from file: " << filename << endl;
    }
    cout << "Success to load camera intrinsics from file: " << filename << endl;
    fs.release();
}

// 加载图片
void loadImage(string const &filepath, cv::Mat &img)
{
    img = cv::imread(filepath);
    Size image_size;
    image_size.width = img.cols;
    image_size.height = img.rows;
    cout << "image_size.width = " << image_size.width << endl;
    cout << "image_size.height = " << image_size.height << endl;
}

//转为灰度图
void conver_BGR2GRAY(cv::Mat &input, cv::Mat &gray, string const &save_path)
{
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    cout << "Success convert image to gray" << endl;
    cv::imwrite(save_path + "/gary.jpg", gray);
}

//寻找角点
void findCorners(cv::Mat &idea_image, Size const &board_size,
                 vector<cv::Point2f> &image_2fpoints_buf, vector<cv::Point2f> &image_2fpoints_seq)
{
    cv::Mat idea_img = idea_image;
   
    /* 提取角点 */
    if (0 == findChessboardCorners(idea_img, board_size, image_2fpoints_buf))
    {
        cout << "can not find chessboard corners!\n"; //找不到角点
        // exit(1);
    }
    else
    {
        /* 亚像素精确化 */
        find4QuadCornerSubpix(idea_img, image_2fpoints_buf, Size(5, 5)); //对粗提取的角点进行精确化

        // image_2fpoints_seq.push_back(image_2fpoints_buf); //保存亚像素角点
        image_2fpoints_seq.assign(image_2fpoints_buf.begin(), image_2fpoints_buf.end());
        /* 在图像上显示角点位置 */
        drawChessboardCorners(idea_img, board_size, image_2fpoints_buf, true); //用于在图片中标记角点
        string src = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/temp_and_result/";
        imwrite(src + "/plotp.jpg", idea_img);
        int npts = image_2fpoints_seq.size();
        if (board_size.width * board_size.height != npts)
        {
            cout << board_size.width * board_size.height << "  " << npts << endl;
            cout << "Failed to find circular grid or board_size.width * board_size.height != npts";
        }
        cout << "Success find chessboard corners " << endl;
    }
}

//整张图像去畸变
void undistortImg(cv::Mat &img_gray, cv::Mat &image_undistort2, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{

    cv::undistort(img_gray, image_undistort2, cameraMatrix, distCoeffs); //去畸变
    string src = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/temp_and_result/";
    imwrite(src + "/undis.jpg", image_undistort2);
    cout << "Success to undistort" << endl;
}

//角点去畸变
void undistortPoint(cv::Mat &img_gray, vector<cv::Point2f> &image_2fpoints_seq, vector<cv::Point2f> &ideal_image_2fpoints, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{
    // vector<cv::Point2f>(_image_2fpoints_seq.size()).swap(_ideal_image_2fpoints);
    cv::undistortPoints(image_2fpoints_seq, ideal_image_2fpoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);
    for (int i = 0; i != 49; ++i)
    {
        cout << "(" << image_2fpoints_seq[i].x << ", " << image_2fpoints_seq[i].y << ") ==> ("
             << ideal_image_2fpoints[i].x << ", " << ideal_image_2fpoints[i].y << ")" << endl;
        cout << i << endl;
    }
    drawPtsOnImage(ideal_image_2fpoints, img_gray);
}

//打印世界坐标和像素点坐标
void printCoordinate(vector<cv::Point3f>  &world_pt, vector<cv::Point2f>  &image_2fpoints_seq){
    printVector(world_pt);
    cout << endl;
    printVector(image_2fpoints_seq);

    // for (auto const &pt : _world_pt)
    // {
    //     cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")" << "  ";
    // }

}


void projectPoints(vector<cv::Point3f> const &worldPts, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &distCoeffs, cv::Mat &cameraMatrix, vector<cv::Point2f> &imgpts)
{
    cv::projectPoints(worldPts, rvec, tvec, cameraMatrix,
                      distCoeffs, imgpts);

    int npts = worldPts.size();
    cout << "Project Points:" << endl;;
    for (int i = 0; i != npts; ++i)
        cout << "(" << worldPts[i].x << ", " << worldPts[i].y << ", " << worldPts[i].z
                  << ") ==> (" << imgpts[i].x << ", " << imgpts[i].y << ")" << endl;

}

void computeErrors(vector<cv::Point2f> const &imgpts, vector<cv::Point2f> const &real_imgpts, vector<double> &errors)
{
    if (imgpts.empty())
        cout<< "imgpts is empty!" <<endl;


    if (imgpts.size() != real_imgpts.size()) {
        cout << "ProjectPoints: the image points are not equal" <<endl;
    }

    int npts = real_imgpts.size();

    auto dist_l1 = [](cv::Point2f const &p1, cv::Point2f const&p2) {return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);};
    vector<double>(npts, 0).swap(errors);
    for (int i = 0; i != npts; ++i)
        errors[i] = dist_l1(imgpts[i], real_imgpts[i]);

    if (!errors.empty()) {
        double sum = 0;
        auto acc = [&sum](double x) {sum += x;};
        std::for_each(errors.begin(), errors.end(), acc);
        sum /= errors.size();
        double maxerr = *std::max_element(errors.begin(), errors.end());
        cout << "average error: " << sum <<endl;
        cout << "max error: " << maxerr << endl;
    }
}

int main()
{
    //计算以板子中心为原点的各个点的坐标
    vector<cv::Point2f> world_point_base_board = get_world_point_base_board();
    cv::Mat _cameraMatrix;
    cv::Mat _distCoeffs;
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat _image;
    cv::Mat _view_gray;
    cv::Mat _img_thres;
    cv::Mat _image_undistort2 = cv::Mat(720, 1280, CV_8UC1);
    Size _board_size = Size(7, 7);
    vector<float> _board_center;
    _board_center.push_back(0);
    _board_center.push_back(-19.2);
    _board_center.push_back(187.3);

    vector<cv::Point2f> _image_2fpoints_buf;   /* 缓存每幅图像上检测到的角点 */
    vector<cv::Point2f> _image_2fpoints_seq;   /* 保存检测到的所有角点 */
    vector<cv::Point2f> _ideal_image_2fpoints; /* 保存检测到的所有角点 */
    vector<cv::Point2f> _imgpts2f; 
    std::vector<double> _errors;

    vector<cv::Point3f> _world_pt;
    // _ideal_image_2fpoints.swap();
    string in_params_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/temp_and_result/calib_Intrinsics.yaml";
    string calib_out_img_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/calib_imgs/out_images/WIN_20220729_17_09_37_Pro.jpg";
    string save_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/calib_ex/temp_and_result";
    loadIntrinsics(in_params_path, _cameraMatrix, _distCoeffs);
    loadImage(calib_out_img_path, _image);
    conver_BGR2GRAY(_image, _view_gray, save_path);
    // cv::adaptiveThreshold(_view_gray, _img_thres, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, );
    undistortImg(_view_gray, _image_undistort2, _cameraMatrix, _distCoeffs);
    // undistortPoint(_view_gray, _image_2fpoints_seq, _ideal_image_2fpoints, _cameraMatrix, _distCoeffs);
    findCorners(_image_undistort2, _board_size, _image_2fpoints_buf, _image_2fpoints_seq);


    float z = 0;
    int nfeat = world_point_base_board.size();
    if (0 == nfeat)
    {
        cout << "WorldPointsEngine: no template points are given" << endl;
        ;
    }
    float x = _board_center[0];
    float y = _board_center[1];
    for (auto const &pt : world_point_base_board)
    {
        _world_pt.emplace_back(cv::Point3f{pt.x + x, y + pt.y, z});
    }
    
    printCoordinate(_world_pt, _image_2fpoints_seq);

    bool ret = cv::solvePnP(_world_pt, _image_2fpoints_seq, _cameraMatrix, _distCoeffs,
                            _rvec, _tvec, false);

    cout << _tvec << endl;

    projectPoints(_world_pt, _rvec, _tvec, _distCoeffs, _cameraMatrix, _imgpts2f);

    computeErrors(_imgpts2f, _image_2fpoints_seq, _errors);

    return 0;
}
