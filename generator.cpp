// Interactive ArUco marker generator with GUI and hotkeys

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

using namespace cv;

struct DictInfo {
    std::string name;
    aruco::PREDEFINED_DICTIONARY_NAME id;
};

static const std::vector<DictInfo> kDicts = {
    {"DICT_4X4_50",       aruco::DICT_4X4_50},
    {"DICT_4X4_100",      aruco::DICT_4X4_100},
    {"DICT_4X4_250",      aruco::DICT_4X4_250},
    {"DICT_4X4_1000",     aruco::DICT_4X4_1000},
    {"DICT_5X5_50",       aruco::DICT_5X5_50},
    {"DICT_5X5_100",      aruco::DICT_5X5_100},
    {"DICT_5X5_250",      aruco::DICT_5X5_250},
    {"DICT_5X5_1000",     aruco::DICT_5X5_1000},
    {"DICT_6X6_50",       aruco::DICT_6X6_50},
    {"DICT_6X6_100",      aruco::DICT_6X6_100},
    {"DICT_6X6_250",      aruco::DICT_6X6_250},
    {"DICT_6X6_1000",     aruco::DICT_6X6_1000},
    {"DICT_7X7_50",       aruco::DICT_7X7_50},
    {"DICT_7X7_100",      aruco::DICT_7X7_100},
    {"DICT_7X7_250",      aruco::DICT_7X7_250},
    {"DICT_7X7_1000",     aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", aruco::DICT_ARUCO_ORIGINAL},
};

struct State {
    int dictIdx = 8;   // default DICT_6X6_50
    int markerId = 0;
    int markerSize = 300; // px (smaller default)
    int borderBits = 1;   // typical 1
    std::string defaultOut; // optional default save path
    bool showHelp = true;
};

static inline Ptr<aruco::Dictionary> currentDictionary(const State& s) {
    int idx = std::max(0, std::min((int)kDicts.size() - 1, s.dictIdx));
    return aruco::getPredefinedDictionary(kDicts[idx].id);
}

static inline int currentDictSize(const State& s) {
    auto dict = currentDictionary(s);
    return dict->bytesList.rows; // number of markers available
}

static Mat renderMarker(const State& s) {
    Mat img;
    auto dict = currentDictionary(s);
    int maxId = std::max(1, currentDictSize(s)) - 1;
    int id = std::max(0, std::min(maxId, s.markerId));
    int bb = std::max(0, std::min(7, s.borderBits));
    int ms = std::max(50, s.markerSize);
    aruco::drawMarker(dict, id, ms, img, bb);

    // Place the marker on a white background with padding so it doesn't touch edges.
    // Increased padding to make the white background larger.
    int margin = std::max(30, ms / 5);
    Mat bg(img.rows + 2*margin, img.cols + 2*margin, CV_8UC1, Scalar(255));
    img.copyTo(bg(Rect(margin, margin, img.cols, img.rows)));
    return bg;
}

static void overlayInfo(Mat& canvas, const State& s, int yStart) {
    // Ensure 3-channel canvas for colored text
    if (canvas.channels() == 1) cvtColor(canvas, canvas, COLOR_GRAY2BGR);

    int baseline = 0;
    double fs = 0.5;     // smaller font size
    int thickness = 1;
    const int lh = 20;   // slightly tighter line spacing
    Point org(10, yStart + 20);

    auto put = [&](const std::string& text, const Scalar& color = Scalar(0,255,0)) {
        putText(canvas, text, org, FONT_HERSHEY_SIMPLEX, fs, Scalar(0,0,0), thickness+2, LINE_AA);
        putText(canvas, text, org, FONT_HERSHEY_SIMPLEX, fs, color, thickness, LINE_AA);
        org.y += lh;
    };

    const auto& dictName = kDicts[std::max(0, std::min((int)kDicts.size()-1, s.dictIdx))].name;
    int maxId = std::max(1, currentDictSize(s)) - 1;
    put("ArUco Marker Generator (GUI)", Scalar(255,255,255));
    put("Dict: " + dictName + "  (d/D prev/next)");
    put(cv::format("ID: %d / %d  (Left/Right; r=random)", std::max(0, std::min(maxId, s.markerId)), maxId));
    put(cv::format("Size: %d px  (Up/Down)", std::max(50, s.markerSize)));
    put(cv::format("Border: %d  ([/])", std::max(0, std::min(7, s.borderBits))));
    if (!s.defaultOut.empty()) put("Save: s -> " + s.defaultOut);
    else put("Save: s -> auto name in CWD");

    if (s.showHelp) {
        org.y += lh/2;
        put("Keys: Left/Right ID  | Up/Down Size  | [/ ] Border", Scalar(200,200,200));
        put("      d/D Prev/Next Dict | r Random ID | s Save PNG", Scalar(200,200,200));
        put("      h Toggle Help | q/ESC Quit", Scalar(200,200,200));
    }
}

static std::string autoFileName(const State& s) {
    const auto& dictName = kDicts[std::max(0, std::min((int)kDicts.size()-1, s.dictIdx))].name;
    return cv::format("marker_%s_id%d_%dpx_bb%d.png", dictName.c_str(), s.markerId,
                      std::max(50, s.markerSize), std::max(0, std::min(7, s.borderBits)));
}

int main(int argc, char** argv) {
    // Optional CLI to set initial state
    const char* keys =
        "{o  |       | default output path for 's' key }"
        "{d  | 8     | dictionary index (0..16, see source) }"
        "{id | 0     | initial marker id }"
        "{ms | 300   | marker size (px) }"  // smaller default
        "{bb | 1     | border bits (0..7) }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Interactive ArUco marker generator with GUI");

    State s;
    if (parser.check()) {
        s.defaultOut = parser.get<String>("o");
        s.dictIdx    = parser.get<int>("d");
        s.markerId   = parser.get<int>("id");
        s.markerSize = parser.get<int>("ms");
        s.borderBits = parser.get<int>("bb");
    } else {
        parser.printErrors();
    }

    srand((unsigned)time(nullptr));

    const std::string kWin = "ArUco Marker";
    namedWindow(kWin, WINDOW_AUTOSIZE);

    bool needRedraw = true;

    for (;;) {
        if (needRedraw) {
            Mat img = renderMarker(s);
            Mat imgBgr;
            if (img.channels() == 1) cvtColor(img, imgBgr, COLOR_GRAY2BGR); else imgBgr = img;

            // Compute info panel height based on lines
            const int lh = 20; // must match overlayInfo
            int baseLines = 6; // title + dict + id + size + border + save
            int extra = s.showHelp ? 4 : 0; // half-gap + 3 help lines ~ 4 lines
            int infoHeight = 10 + (baseLines + extra) * lh + 10; // padding

            Mat canvas(imgBgr.rows + infoHeight, imgBgr.cols, CV_8UC3, Scalar(255,255,255));
            // place marker on top
            imgBgr.copyTo(canvas(Rect(0, 0, imgBgr.cols, imgBgr.rows)));
            // draw info below
            overlayInfo(canvas, s, imgBgr.rows);
            imshow(kWin, canvas);
            needRedraw = false;
        }

    int key = waitKeyEx(0); // extended keys (arrows, etc.)
    if (key < 0) continue;

    // Interpret ASCII only for plain keys (<= 255). Do NOT mask extended codes,
    // otherwise arrows like Left (0xFF51) would become 'Q' and trigger quit.
    const bool isCharKey = (key >= 0 && key <= 255);
    const int ch = isCharKey ? key : -1;

    // Quit (ASCII)
    if (ch == 27 || ch == 'q' || ch == 'Q') break;

        // Toggle help
    if (ch == 'h' || ch == 'H') { s.showHelp = !s.showHelp; needRedraw = true; continue; }

        // Dictionary cycle: 'd' previous, 'D' next
    if (ch == 'd') { s.dictIdx = (s.dictIdx - 1 + (int)kDicts.size()) % (int)kDicts.size();
                          // clamp id to available range
                          s.markerId = std::min(s.markerId, currentDictSize(s) - 1);
                          needRedraw = true; continue; }
    if (ch == 'D') { s.dictIdx = (s.dictIdx + 1) % (int)kDicts.size();
                          s.markerId = std::min(s.markerId, currentDictSize(s) - 1);
                          needRedraw = true; continue; }

        // Arrow key codes:
        // Linux/X11: Left=65361, Up=65362, Right=65363, Down=65364
        // Windows:   Left=2424832, Up=2490368, Right=2555904, Down=2621440
        const int KEY_LEFT_1 = 65361, KEY_LEFT_2 = 2424832;
        const int KEY_RIGHT_1 = 65363, KEY_RIGHT_2 = 2555904;
        const int KEY_UP_1 = 65362, KEY_UP_2 = 2490368;
        const int KEY_DOWN_1 = 65364, KEY_DOWN_2 = 2621440;

        // ID adjust: Left/Right arrows or ',' '.' keys
    if (key == KEY_LEFT_1 || key == KEY_LEFT_2 || ch == ',' ) {
            s.markerId = (s.markerId - 1);
            if (s.markerId < 0) s.markerId = currentDictSize(s) - 1;
            needRedraw = true; continue;
        }
    if (key == KEY_RIGHT_1 || key == KEY_RIGHT_2 || ch == '.' ) {
            s.markerId = (s.markerId + 1) % std::max(1, currentDictSize(s));
            needRedraw = true; continue;
        }

        // Size adjust: Up/Down arrows
        if (key == KEY_UP_1 || key == KEY_UP_2) { s.markerSize = std::min(4096, s.markerSize + 50); needRedraw = true; continue; }
        if (key == KEY_DOWN_1 || key == KEY_DOWN_2) { s.markerSize = std::max(50, s.markerSize - 50); needRedraw = true; continue; }

        // Border bits adjust: '[' ']' keys
    if (ch == '[') { s.borderBits = std::max(0, s.borderBits - 1); needRedraw = true; continue; }
    if (ch == ']') { s.borderBits = std::min(7, s.borderBits + 1); needRedraw = true; continue; }

        // Random ID
    if (ch == 'r' || ch == 'R') {
            int maxId = std::max(1, currentDictSize(s));
            s.markerId = rand() % maxId;
            needRedraw = true; continue;
        }

        // Save PNG
    if (ch == 's' || ch == 'S') {
            Mat img = renderMarker(s);
            std::string path = s.defaultOut.empty() ? autoFileName(s) : s.defaultOut;
            try {
                imwrite(path, img);
                // Briefly flash a message by redrawing overlay
                Mat imgBgr; if (img.channels()==1) cvtColor(img, imgBgr, COLOR_GRAY2BGR); else imgBgr = img;
                const int lh = 20; int baseLines = 6; int extra = s.showHelp ? 4 : 0;
                int infoHeight = 10 + (baseLines + extra) * lh + 10;
                Mat canvas(imgBgr.rows + infoHeight, imgBgr.cols, CV_8UC3, Scalar(255,255,255));
                imgBgr.copyTo(canvas(Rect(0,0,imgBgr.cols,imgBgr.rows)));
                overlayInfo(canvas, s, imgBgr.rows);
                putText(canvas, "Saved: " + path, Point(10, canvas.rows - 10), FONT_HERSHEY_SIMPLEX,
                        0.6, Scalar(0,0,0), 3, LINE_AA);
                putText(canvas, "Saved: " + path, Point(10, canvas.rows - 10), FONT_HERSHEY_SIMPLEX,
                        0.6, Scalar(0,255,255), 2, LINE_AA);
                imshow(kWin, canvas);
            } catch (const std::exception&){ /* ignore */ }
            continue;
        }

        // Any other key: ignore
    }

    destroyAllWindows();
    return 0;
}