#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <chrono>
#if defined(__unix__) || defined(__APPLE__)
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include <signal.h> // signals for clean exit

using namespace std;
using namespace cv;
using namespace cv::aruco;

// Replace CLOCK()/avg* helpers with clearer chrono-based utilities
static inline double nowMs() {
    using clock = std::chrono::steady_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

struct FpsStats {
    double avgMs = 0.0;
    double fpsStart = nowMs();
    double avgFps = 0.0;
    double fps1sec = 0.0;

    double updateAvgMs(double frameMs) {
        avgMs = 0.98 * avgMs + 0.02 * frameMs;
        return avgMs;
    }
    double tickFps() {
        double now = nowMs();
        if (now - fpsStart > 1000.0) {
            fpsStart = now;
            avgFps = 0.7 * avgFps + 0.3 * fps1sec;
            fps1sec = 0.0;
        }
        fps1sec += 1.0;
        return avgFps;
    }
};
// ---- End timing helpers ----

// ---- Terminal (Unix) non-blocking input helpers ----
#if defined(__unix__) || defined(__APPLE__)
static struct termios orig_termios;
static bool term_raw_enabled = false;

void enableRawTerminal() {
    if (term_raw_enabled) return;
    struct termios raw;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    term_raw_enabled = true;
}
void disableRawTerminal() {
    if (!term_raw_enabled) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    term_raw_enabled = false;
}
bool stdinKeyPressed(int& ch) {
    unsigned char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n == 1) { ch = c; return true; }
    return false;
}
#else
void enableRawTerminal() {}
void disableRawTerminal() {}
bool stdinKeyPressed(int&) { return false; }
#endif
// ---- End terminal helpers ----

// Ensure raw terminal is restored automatically
struct TerminalRawGuard {
    TerminalRawGuard() { enableRawTerminal(); }
    ~TerminalRawGuard() { disableRawTerminal(); }
};

// Add exit key matcher and signal handling
static inline bool isExitKey(int k) {
    if (k < 0) return false;
    k &= 0xFF; // normalize
    switch (k) {
        case 27:            // ESC
        case 'q': case 'Q': // quit
        case 'x': case 'X': // exit
        case 'c': case 'C': // close
        case 3:             // Ctrl+C
        case 4:             // Ctrl+D
        case 17:            // Ctrl+Q
        case 24:            // Ctrl+X
            return true;
        default:
            return false;
    }
}

volatile sig_atomic_t g_signal_exit = 0;
void handleSignal(int) { g_signal_exit = 1; }

// Centralized exit request check (window key, terminal key, or signal)
static inline bool exitRequested(int windowKey) {
    if (isExitKey(windowKey)) return true;
#if defined(__unix__) || defined(__APPLE__)
    int ch;
    if (stdinKeyPressed(ch) && isExitKey(ch)) return true;
#endif
    if (g_signal_exit) return true;
    return false;
}

int main() {
    // Constants for clarity
    const int kFrameWidth = 640;
    const int kFrameHeight = 480;
    const char* kWindowTitle = "Aruco Detect";

    // 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 设置分辨率 640x480
    cap.set(cv::CAP_PROP_FRAME_WIDTH, kFrameWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kFrameHeight);

    // 使用 OpenCV 自带的 ArUco 字典
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);

    Mat frame;

    // Enable terminal key handling with RAII
    TerminalRawGuard terminalGuard;

    // Register signal handlers for clean exit (restores terminal)
    signal(SIGINT, handleSignal);
    signal(SIGTERM, handleSignal);
    signal(SIGHUP, handleSignal);

    FpsStats stats;

    while (true) {
        double start = nowMs(); // start timing this frame

        if (!cap.read(frame) || frame.empty()) break;

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        // 检测 ArUco marker
        cv::aruco::detectMarkers(frame, dictionary, corners, ids);

        // 如果检测到 marker 就画出来
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(frame, corners, ids);
            // for (int id : ids) std::cout << "检测到 ID: " << id << std::endl;
        }

        // Overlay averaged duration and FPS
        double dur = nowMs() - start; // ms
        std::string statsText = cv::format("avg %.2f ms  fps %.1f",
                                           stats.updateAvgMs(dur), stats.tickFps());
        cv::putText(frame, statsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

        cv::imshow(kWindowTitle, frame);

        // Unified exit handling
        int key = cv::waitKey(1);
        if (exitRequested(key)) break;
    }

    // Terminal restored automatically by TerminalRawGuard
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

