#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>

#pragma comment(lib,"ws2_32.lib")

using namespace std;

struct Detection
{
    int x;
    int y;
    int w;
    int h;
    int cls;
    float conf;
};

vector<string> class_names =
{
    "buff_chicken_leg",
    "buff_powerup",
    "chicken_boss",
    "chicken_minion",
    "egg",
    "feathers",
    "laser",
    "metalic_object",
    "meteor",
    "ship"
};

vector<cv::Scalar> class_colors =
{
    {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},
    {0,255,255},{128,0,255},{0,128,255},{255,128,0},{0,255,128}
};

cv::Mat captureScreen(int x,int y,int width,int height)
{
    HDC hScreen = GetDC(NULL);
    HDC hdcMem = CreateCompatibleDC(hScreen);

    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen,width,height);
    SelectObject(hdcMem,hBitmap);

    BitBlt(hdcMem,0,0,width,height,hScreen,x,y,SRCCOPY);

    BITMAPINFOHEADER bi{};
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    cv::Mat mat(height,width,CV_8UC3);

    GetDIBits(
        hdcMem,
        hBitmap,
        0,
        height,
        mat.data,
        (BITMAPINFO*)&bi,
        DIB_RGB_COLORS
    );

    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(NULL,hScreen);

    return mat;
}

string detectionsToJson(vector<Detection>& dets)
{
    stringstream ss;
    ss << "[";

    for(size_t i=0;i<dets.size();i++)
    {
        auto &d = dets[i];

        ss << "{";
        ss << "\"cls\":" << d.cls << ",";
        ss << "\"x\":" << d.x << ",";
        ss << "\"y\":" << d.y << ",";
        ss << "\"w\":" << d.w << ",";
        ss << "\"h\":" << d.h << ",";
        ss << "\"conf\":" << d.conf;
        ss << "}";

        if(i < dets.size()-1) ss << ",";
    }

    ss << "]";
    return ss.str();
}

int main()
{
    // SOCKET INIT
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2),&wsa);

    SOCKET sock = socket(AF_INET,SOCK_STREAM,0);

    sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(5000);
    inet_pton(AF_INET,"127.0.0.1",&server.sin_addr);

    connect(sock,(sockaddr*)&server,sizeof(server));

    // MODEL
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"model");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env,L"tavuk.onnx",session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_ptr = session.GetInputNameAllocated(0,allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0,allocator);

    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    const int INPUT_SIZE = 640;

    vector<int64_t> input_shape = {1,3,INPUT_SIZE,INPUT_SIZE};
    size_t input_tensor_size = 1*3*INPUT_SIZE*INPUT_SIZE;

    vector<float> input_tensor_values(input_tensor_size);

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);

    int top=100,left=100,width=700,height=500;

    vector<Detection> last_detections;

    while(true)
    {
        cv::Mat frame = captureScreen(left,top,width,height);

        last_detections.clear();

        cv::Mat resized;
        cv::resize(frame,resized,cv::Size(INPUT_SIZE,INPUT_SIZE));

        int index=0;

        for(int c=2;c>=0;c--)
        for(int y=0;y<INPUT_SIZE;y++)
        {
            const uchar* row=resized.ptr<uchar>(y);

            for(int x=0;x<INPUT_SIZE;x++)
            {
                input_tensor_values[index++] = row[x*3+c]/255.0f;
            }
        }

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_size,
                input_shape.data(),
                input_shape.size()
            );

        auto output_tensors =
            session.Run(
                Ort::RunOptions{nullptr},
                &input_name,
                &input_tensor,
                1,
                &output_name,
                1
            );

        float* output =
            output_tensors[0].GetTensorMutableData<float>();

        int detections=300;
        int values=14;
        int num_classes=10;

        for(int i=0;i<detections;i++)
        {
            float cx=output[i*values+0];
            float cy=output[i*values+1];
            float bw=output[i*values+2];
            float bh=output[i*values+3];

            float max_score=0;
            int max_class=0;

            for(int c=0;c<num_classes;c++)
            {
                float score=output[i*values+4+c];

                if(score>max_score)
                {
                    max_score=score;
                    max_class=c;
                }
            }

            if(max_score<0.5) continue;

            Detection d;

            d.x=(cx-bw/2)*width;
            d.y=(cy-bh/2)*height;
            d.w=bw*width;
            d.h=bh*height;
            d.cls=max_class;
            d.conf=max_score;

            last_detections.push_back(d);
        }

        // PYTHON'A GÖNDER
        string json = detectionsToJson(last_detections);
        send(sock,json.c_str(),json.size(),0);

        cv::imshow("Detection",frame);
        if(cv::waitKey(1)=='q') break;
    }

    closesocket(sock);
    WSACleanup();

    return 0;
}
