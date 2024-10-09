#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

// Function to add Gaussian noise to an image
Mat addnoise(Mat x) {
    Mat noiseimg;
    Mat img2;
    img2.create(x.rows, x.cols, x.type());
    noiseimg.create(x.rows, x.cols, x.type());
    randn(noiseimg, 0, 100); // Gaussian noise
    add(x, noiseimg, img2);
    return img2;
}

Mat cummulativehis(Mat img) {
    Mat imggray;
    cvtColor(img, imggray, COLOR_BGR2GRAY);

    int histogram[256] = { 0 }; // initialize all intensity values to 0

    // calculate the no of pixels for each intensity value
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            histogram[(int)imggray.at<uchar>(y, x)]++;
        }
    }

    // calculate the cumulative histogram
    int cumhistogram[256];
    cumhistogram[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cumhistogram[i] = histogram[i] + cumhistogram[i - 1];
    }

    int cumhalf = (cumhistogram[255] / 100) * 70;
    int detguassval = 0;

    cout << "70%: " << cumhalf << endl;

    for (int i = 0; i < 256; i++) {
        if (cumhistogram[i] > cumhalf) {
            detguassval = i;
            break;
        }
    }

    cout << "Cummulative histogram value: " << detguassval << endl;

    imshow("Gray Image", imggray);
    return imggray;
}

double estimateNoiseLevel(Mat src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = src;
    }

    Mat laplacian;
    Laplacian(gray, laplacian, CV_64F);

    Scalar mean, stddev;
    meanStdDev(laplacian, mean, stddev);

    return stddev[0]; // Return the standard deviation as the noise level
}

Mat opencvgb(Mat image, int x) {
    Mat kernelimage;
    GaussianBlur(image, kernelimage, Size(x, x), 0);
    return kernelimage;
}

int main() {
    // Read the image from the specified path
    //Mat img = imread("C:/Users//User/Desktop/projectpic/00011331/billshire_36.jpg");
     Mat img = imread("C:/Users/user/Pictures/onepic.png");
    
    // Check if the image is loaded successfully
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Add noise to the image
    Mat noisyImg = addnoise(img);

    // Calculate the Laplacian (noise level)
    double laplacianValue = estimateNoiseLevel(noisyImg);
    cout << "Laplacian Value (Noise Level): " << laplacianValue << endl;

    // If Laplacian value is 80 and above, apply Gaussian blur after cumulative histogram
    if (laplacianValue >= 80) {
        cout << "Laplacian is 80 or above, applying cumulative histogram and Gaussian blur." << endl;

        // Apply cumulative histogram
        Mat cumnoisy = cummulativehis(noisyImg);

        // Apply Gaussian blur with different kernel sizes
        int kernelsizes[] = { 3, 5, 7, 9 };
        for (int kernelsize : kernelsizes) {
            cout << "Kernel size: " << kernelsize << endl;
            Mat gb = opencvgb(noisyImg, kernelsize);

            // Apply cumulative histogram to the blurred image
            Mat gbresult = cummulativehis(gb);
        }
    }
    else {
        cout << "Laplacian is below 80, only applying cumulative histogram." << endl;

        // Apply cumulative histogram only
        Mat cumnoisy = cummulativehis(noisyImg);
    }

    // Wait for a keystroke in the window
    waitKey(0);
    return 0;
}
