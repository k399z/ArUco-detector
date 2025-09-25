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

    // Prepare a list of all ArUco dictionaries we want to detect
    struct DictInfo { const char* name; cv::aruco::PREDEFINED_DICTIONARY_NAME id; };
    static const std::vector<DictInfo> kDicts = {
        {"4X4_50",        cv::aruco::DICT_4X4_50},
        {"4X4_100",       cv::aruco::DICT_4X4_100},
        {"4X4_250",       cv::aruco::DICT_4X4_250},
        {"4X4_1000",      cv::aruco::DICT_4X4_1000},
        {"5X5_50",        cv::aruco::DICT_5X5_50},
        {"5X5_100",       cv::aruco::DICT_5X5_100},
        {"5X5_250",       cv::aruco::DICT_5X5_250},
        {"5X5_1000",      cv::aruco::DICT_5X5_1000},
        {"6X6_50",        cv::aruco::DICT_6X6_50},
        {"6X6_100",       cv::aruco::DICT_6X6_100},
        {"6X6_250",       cv::aruco::DICT_6X6_250},
        {"6X6_1000",      cv::aruco::DICT_6X6_1000},
        {"7X7_50",        cv::aruco::DICT_7X7_50},
        {"7X7_100",       cv::aruco::DICT_7X7_100},
        {"7X7_250",       cv::aruco::DICT_7X7_250},
        {"7X7_1000",      cv::aruco::DICT_7X7_1000},
        {"ARUCO_ORIGINAL",cv::aruco::DICT_ARUCO_ORIGINAL},
    };
    std::vector<cv::Ptr<cv::aruco::Dictionary>> dictionaries;
    dictionaries.reserve(kDicts.size());
    for (const auto& d : kDicts) {
        dictionaries.emplace_back(cv::aruco::getPredefinedDictionary(d.id));
    }
    // Tune detection parameters slightly for better recall on small markers
    cv::Ptr<cv::aruco::DetectorParameters> detParams = cv::aruco::DetectorParameters::create();
    detParams->adaptiveThreshWinSizeMin = 3;
    detParams->adaptiveThreshWinSizeMax = 23; // keep small for speed
    detParams->adaptiveThreshWinSizeStep = 10;
    detParams->minMarkerPerimeterRate = 0.03f; // detect smaller markers
    detParams->maxMarkerPerimeterRate = 4.0f;
    detParams->polygonalApproxAccuracyRate = 0.03;
    detParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE; // faster

    // Runtime tunables for FPS on low-power devices
    double detectScale = 0.5;     // downscale for detection (0.33..1.0)
    int detectEveryN = 1;         // detect every N frames (1=no skip)
    bool useAllDicts = false;     // toggle to scan all dictionaries
    // A practical fast subset: 6x6_250 and ARUCO_ORIGINAL
    std::vector<int> subsetIdx = {10, 16}; // indices in kDicts

    Mat frame;
    Mat gray, detectImg;

    // Cache last detections when using stride >1
    std::vector<int> lastIds;
    std::vector<std::vector<cv::Point2f>> lastCorners;
    std::vector<std::string> lastLabels;
    int frameIndex = 0;

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

        // Prepare grayscale and optionally downscaled image for faster detection
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (detectScale < 0.999) {
            cv::resize(gray, detectImg, Size(), detectScale, detectScale, cv::INTER_AREA);
        } else {
            detectImg = gray;
        }

        // Accumulate detections from all dictionaries
        std::vector<int> allIds; allIds.reserve(64);
        std::vector<std::vector<cv::Point2f>> allCorners; allCorners.reserve(64);
        std::vector<std::string> labels; labels.reserve(64);

        bool doDetect = ((frameIndex % detectEveryN) == 0);
        if (doDetect) {
            auto detectInDict = [&](size_t i){
                std::vector<int> ids;
                std::vector<std::vector<cv::Point2f>> corners;
                cv::aruco::detectMarkers(detectImg, dictionaries[i], corners, ids, detParams);
                if (!ids.empty()) {
                    double inv = detectScale < 0.999 ? (1.0/detectScale) : 1.0;
                    for (size_t k = 0; k < ids.size(); ++k) {
                        if (detectScale < 0.999) {
                            for (auto& p : corners[k]) { p.x = float(p.x * inv); p.y = float(p.y * inv); }
                        }
                        allCorners.push_back(corners[k]);
                        allIds.push_back(ids[k]);
                        labels.emplace_back(std::string(kDicts[i].name) + ":" + std::to_string(ids[k]));
                    }
                }
            };

            if (useAllDicts) {
                for (size_t i = 0; i < dictionaries.size(); ++i) detectInDict(i);
            } else {
                for (int idx : subsetIdx) if (idx >=0 && idx < (int)dictionaries.size()) detectInDict((size_t)idx);
            }

            // Store for skipped frames
            lastIds = allIds; lastCorners = allCorners; lastLabels = labels;
        } else {
            allIds = lastIds; allCorners = lastCorners; labels = lastLabels;
        }

        // Draw all detected markers
        if (!allIds.empty()) {
            cv::aruco::drawDetectedMarkers(frame, allCorners, allIds);
            // Put dictionary label near each marker center
            for (size_t i = 0; i < allCorners.size(); ++i) {
                const auto& pts = allCorners[i];
                cv::Point2f c(0,0);
                for (const auto& p : pts) c += p; c *= (1.0f/4.0f);
                cv::putText(frame, labels[i], c + cv::Point2f(-20, -10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 2, cv::LINE_AA);
            }
        }

        // Overlay averaged duration, FPS, and detection count (+ mode info)
        double dur = nowMs() - start; // ms
        std::string statsText = cv::format("avg %.2f ms  fps %.1f  det %d  mode %s  scale %.2f  stride %d",
                                           stats.updateAvgMs(dur), stats.tickFps(), (int)labels.size(),
                                           useAllDicts ? "ALL" : "FAST", detectScale, detectEveryN);
        cv::putText(frame, statsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

        cv::imshow(kWindowTitle, frame);

        // Unified exit handling
        int key = cv::waitKey(1);
        // Simple hotkeys for performance tuning
        if (key >= 0) {
            int k = key & 0xFF;
            if (k == 'a' || k == 'A') useAllDicts = !useAllDicts;
            if (k == '1') detectEveryN = 1;
            if (k == '2') detectEveryN = 2;
            if (k == '3') detectEveryN = 3;
            if (k == '4') detectEveryN = 4;
            if (k == 'z' || k == 'Z') {
                // cycle common scales: 1.0 -> 0.75 -> 0.5 -> 0.33 -> 1.0
                const double scales[] = {1.0, 0.75, 0.5, 0.33};
                for (int i=0;i<4;i++) if (fabs(detectScale - scales[i]) < 1e-3) { detectScale = scales[(i+1)%4]; break; }
            }
        }
        if (exitRequested(key)) break;
        ++frameIndex;
    }

    // Terminal restored automatically by TerminalRawGuard
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

