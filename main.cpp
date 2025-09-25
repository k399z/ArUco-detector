#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main() {
    // 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 设置分辨率 640x480
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 使用 OpenCV 自带的 ArUco 字典
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        // 检测 ArUco marker
        cv::aruco::detectMarkers(frame, dictionary, corners, ids);

        // 如果检测到 marker 就画出来
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(frame, corners, ids);
            for (size_t i = 0; i < ids.size(); i++) {
                std::cout << "检测到 ID: " << ids[i] << std::endl;
            }
        }

        cv::imshow("Aruco Detect", frame);

        // 按 ESC 退出
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

