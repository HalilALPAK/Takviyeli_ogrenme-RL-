#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <sstream>

#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")

using namespace std;

// Grid düzeyinde hareket için güncellenmiş fonksiyon
void simulateKey(WORD vKey) {
    if (vKey == 0) return; 
    INPUT input = {0};
    input.type = INPUT_KEYBOARD;
    input.ki.wScan = MapVirtualKey(vKey, MAPVK_VK_TO_VSC);
    input.ki.dwFlags = KEYEVENTF_SCANCODE;
    if (vKey == VK_LEFT || vKey == VK_RIGHT || vKey == VK_UP || vKey == VK_DOWN) 
        input.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;

    // Tuşa Bas
    SendInput(1, &input, sizeof(INPUT));
    
    // 130ms: Geminin yaklaşık 1 grid hücresi kayması için gereken süre
    // Oyun hızına göre 120-150 arası optimize edilebilir.
    Sleep(130); 

    // Tuşu Bırak
    input.ki.dwFlags |= KEYEVENTF_KEYUP;
    SendInput(1, &input, sizeof(INPUT));
}

// captureScreen ve getClsColor fonksiyonları öncekiyle aynı kalıyor...
cv::Scalar getClsColor(int cls) {
    switch (cls) {
        case 9: return cv::Scalar(0, 255, 0);   
        case 3: return cv::Scalar(0, 0, 255);   
        case 4: return cv::Scalar(0, 255, 255); 
        case 8: return cv::Scalar(0, 165, 255); 
        default: return cv::Scalar(255, 0, 255); 
    }
}

cv::Mat captureScreen(int x, int y, int width, int height) {
    HDC hScreen = GetDC(NULL); HDC hdcMem = CreateCompatibleDC(hScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, width, height);
    SelectObject(hdcMem, hBitmap); BitBlt(hdcMem, 0, 0, width, height, hScreen, x, y, SRCCOPY);
    BITMAPINFOHEADER bi{ sizeof(BITMAPINFOHEADER), width, -height, 1, 24, BI_RGB };
    cv::Mat mat(height, width, CV_8UC3);
    GetDIBits(hdcMem, hBitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    DeleteObject(hBitmap); DeleteDC(hdcMem); ReleaseDC(NULL, hScreen);
    return mat;
}

int main() {
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in server; server.sin_family = AF_INET; server.sin_port = htons(5001);
    inet_pton(AF_INET, "127.0.0.1", &server.sin_addr);

    cout << "Python bekleniyor..." << endl;
    while (connect(sock, (sockaddr*)&server, sizeof(server)) == SOCKET_ERROR) { Sleep(500); }
    cout << "Baglandi! Grid Hareketi Aktif." << endl;
    u_long iMode = 1; ioctlsocket(sock, FIONBIO, &iMode);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"tavuk.onnx", session_options);
    auto allocator = Ort::AllocatorWithDefaultOptions();
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    const char* in_names[] = { in_name.get() }; const char* out_names[] = { out_name.get() };

    int left = 100, top = 100, width = 700, height = 500;
    int cellW = width / 10, cellH = height / 8;

    while (true) {
        char recvBuf[16];
        int res = recv(sock, recvBuf, sizeof(recvBuf), 0);
        if (res > 0) {
            char cmd = recvBuf[0];
            if (cmd == 'U') simulateKey(VK_UP);
            else if (cmd == 'D') simulateKey(VK_DOWN);
            else if (cmd == 'L') simulateKey(VK_LEFT);
            else if (cmd == 'R') simulateKey(VK_RIGHT);
            else if (cmd == 'S') simulateKey(VK_SPACE);
        }

        cv::Mat frame = captureScreen(left, top, width, height);
        cv::Mat debugFrame = frame.clone();
        cv::Mat resized; cv::resize(frame, resized, {640,640});
        vector<float> input_tensor(1*3*640*640);
        for(int c=0; c<3; c++) for(int y=0; y<640; y++) for(int x=0; x<640; x++)
            input_tensor[c*640*640 + y*640 + x] = resized.at<cv::Vec3b>(y,x)[2-c]/255.0f;

        int64_t shape[] = {1, 3, 640, 640};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_t = Ort::Value::CreateTensor<float>(mem, input_tensor.data(), input_tensor.size(), shape, 4);
        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_t, 1, out_names, 1);
        float* out = outputs[0].GetTensorMutableData<float>();

        stringstream ss;
        ss << "{\"d\":[";
        bool first = true;
        for(int i=0; i<300; i++) {
            float conf = 0; int cls = -1;
            for(int c=0; c<10; c++) if(out[i*14+4+c] > conf) { conf=out[i*14+4+c]; cls=c; }
            if(conf > 0.45) {
                float cx = out[i*14+0] * width, cy = out[i*14+1] * height;
                float w = out[i*14+2] * width, h = out[i*14+3] * height;
                if(!first) ss << ",";
                ss << "{\"c\":" << cls << ",\"x\":" << (int)(cx/cellW) << ",\"y\":" << (int)(cy/cellH) << "}";
                first = false;
                cv::rectangle(debugFrame, cv::Rect(cx-w/2, cy-h/2, w, h), getClsColor(cls), 2);
            }
        }
        ss << "]}\n";
        send(sock, ss.str().c_str(), (int)ss.str().length(), 0);
        cv::imshow("AI Vision Debug", debugFrame);
        if(cv::waitKey(1) == 'q') break;
    }
    return 0;
}