#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

// Shared memory structure
struct SharedData {
    int imageWidth;
    int imageHeight;
    int imageChannels;
    int numBoxes;
    // Followed by image data and then bounding boxes
};

struct BoundingBox {
    int classId;
    float confidence;
    int x;
    int y;
    int width;
    int height;
};

// Constants
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

// COCO class names - index 0 is person
const vector<string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

vector<BoundingBox> detect_humans(Mat& image, Net& net) {
    Mat blob;
    vector<BoundingBox> detections;

    // Create blob from image
    blobFromImage(image, blob, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    // Forward pass
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Process outputs
    float x_factor = image.cols / static_cast<float>(INPUT_WIDTH);
    float y_factor = image.rows / static_cast<float>(INPUT_HEIGHT);

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Parse YOLO output
    for (auto& output : outputs) {
        auto data = (float*)output.data;
        for (int i = 0; i < output.rows; i++) {
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
                float* classes_scores = data + 5;
                Mat scores(1, output.cols - 5, CV_32FC1, classes_scores);
                Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                // Only detect persons (class_id = 0)
                if (max_class_score > CONFIDENCE_THRESHOLD && class_id.x == 0) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = static_cast<int>((cx - 0.5 * w) * x_factor);
                    int top = static_cast<int>((cy - 0.5 * h) * y_factor);
                    int width = static_cast<int>(w * x_factor);
                    int height = static_cast<int>(h * y_factor);

                    boxes.push_back(Rect(left, top, width, height));
                }
            }
            data += output.cols;
        }
    }

    // Apply Non-Maximum Suppression
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_result);

    // Create bounding box structures
    for (int idx : nms_result) {
        BoundingBox box;
        box.classId = class_ids[idx];
        box.confidence = confidences[idx];
        box.x = boxes[idx].x;
        box.y = boxes[idx].y;
        box.width = boxes[idx].width;
        box.height = boxes[idx].height;
        detections.push_back(box);
    }

    return detections;
}

bool writeToSharedMemory(const Mat& image, const vector<BoundingBox>& boxes) {
    // Calculate total size needed
    size_t imageSize = image.total() * image.elemSize();
    size_t boxesSize = boxes.size() * sizeof(BoundingBox);
    size_t totalSize = sizeof(SharedData) + imageSize + boxesSize;

    // Create shared memory
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        totalSize,
        L"YOLODetectionSharedMemory"
    );

    if (hMapFile == NULL) {
        cerr << "Could not create file mapping object: " << GetLastError() << endl;
        return false;
    }

    // Map view of file
    LPVOID pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        totalSize
    );

    if (pBuf == NULL) {
        cerr << "Could not map view of file: " << GetLastError() << endl;
        CloseHandle(hMapFile);
        return false;
    }

    // Write header
    SharedData* header = static_cast<SharedData*>(pBuf);
    header->imageWidth = image.cols;
    header->imageHeight = image.rows;
    header->imageChannels = image.channels();
    header->numBoxes = boxes.size();

    // Write image data
    unsigned char* imagePtr = static_cast<unsigned char*>(pBuf) + sizeof(SharedData);
    memcpy(imagePtr, image.data, imageSize);

    // Write bounding boxes
    BoundingBox* boxPtr = reinterpret_cast<BoundingBox*>(imagePtr + imageSize);
    for (size_t i = 0; i < boxes.size(); i++) {
        boxPtr[i] = boxes[i];
    }

    cout << "Data written to shared memory successfully!" << endl;
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "Number of persons detected: " << boxes.size() << endl;

    // Keep the shared memory alive
    cout << "Press Enter to close shared memory..." << endl;
    cin.get();

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);

    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <image_path> <yolo_model_path>" << endl;
        cout << "Example: " << argv[0] << " person.jpg yolov5s.onnx" << endl;
        return -1;
    }

    string imagePath = argv[1];
    string modelPath = argv[2];

    // Load image
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Error: Could not load image from " << imagePath << endl;
        return -1;
    }

    cout << "Image loaded: " << image.cols << "x" << image.rows << endl;

    // Load YOLO model
    cout << "Loading YOLO model from " << modelPath << "..." << endl;
    Net net;
    try {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    catch (const Exception& e) {
        cerr << "Error loading model: " << e.what() << endl;
        return -1;
    }

    cout << "Model loaded successfully!" << endl;

    // Detect humans
    cout << "Detecting humans..." << endl;
    vector<BoundingBox> detections = detect_humans(image, net);

    cout << "Found " << detections.size() << " person(s) in the image" << endl;

    // Draw bounding boxes on image for visualization
    Mat displayImage = image.clone();
    for (const auto& box : detections) {
        rectangle(displayImage, Rect(box.x, box.y, box.width, box.height), Scalar(0, 255, 0), 2);
        string label = "Person: " + to_string(static_cast<int>(box.confidence * 100)) + "%";
        putText(displayImage, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    // Save result image
    imwrite("detection_result.jpg", displayImage);
    cout << "Result saved to detection_result.jpg" << endl;

    // Write to shared memory
    cout << "\nWriting to shared memory..." << endl;
    if (!writeToSharedMemory(image, detections)) {
        cerr << "Failed to write to shared memory" << endl;
        return -1;
    }

    return 0;
}
