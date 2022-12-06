#include "arcface.h"
#include <math.h>

extern "C" {
    double __acos_finite(double x) { return acos(x); }
    float __acosf_finite(float x)  { return acosf(x); }
    double __acosh_finite(double x) { return acosh(x); }
    float __acoshf_finite(float x)  { return acoshf(x); }
    double __asin_finite(double x) { return asin(x); }
    float __asinf_finite(float x)  { return asinf(x); }
    double __atanh_finite(double x) { return atanh(x); }
    float __atanhf_finite(float x)  { return atanhf(x); }
    double __cosh_finite(double x) { return cosh(x); }
    float __coshf_finite(float x)  { return coshf(x); }
    double __sinh_finite(double x) { return sinh(x); }
    float __sinhf_finite(float x)  { return sinhf(x); }
    double __exp_finite(double x) { return exp(x); }
    float __expf_finite(float x)  { return expf(x); }
    double __log10_finite(double x) { return log10(x); }
    float __log10f_finite(float x)  { return log10f(x); }
    double __log_finite(double x) { return log(x); }
    float __logf_finite(float x)  { return logf(x); }
    double __atan2_finite(double x, double y) { return atan2(x,y); }
    float __atan2f_finite(float x, double y)  { return atan2f(x,y); }
    double __pow_finite(double x, double y) { return pow(x,y); }
    float __powf_finite(float x, double y)  { return powf(x,y); }
    double __remainder_finite(double x, double y) { return remainder(x,y); }
    float __remainderf_finite(float x, double y)  { return remainderf(x,y); }
}

cv::Mat ncnn2cv(ncnn::Mat img)
{
    unsigned char pix[img.h * img.w * 3];
    img.to_pixels(pix, ncnn::Mat::PIXEL_BGR);
    cv::Mat cv_img(img.h, img.w, CV_8UC3);
    for (int i = 0; i < cv_img.rows; i++)
    {
        for (int j = 0; j < cv_img.cols; j++)
        {
            cv_img.at<cv::Vec3b>(i,j)[0] = pix[3 * (i * cv_img.cols + j)];
            cv_img.at<cv::Vec3b>(i,j)[1] = pix[3 * (i * cv_img.cols + j) + 1];
            cv_img.at<cv::Vec3b>(i,j)[2] = pix[3 * (i * cv_img.cols + j) + 2];
        }
    }
    return cv_img;
}

int main(int argc, char* argv[])
{
     // Read Image
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [imagepath1] [imagepath2]\n", argv[0]);
        return -1;
    }

    const char* imagepath1 = argv[1];
    const char* imagepath2 = argv[2];

    cv::Mat m1 = cv::imread(imagepath1, 1);
    if (m1.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath1);
        return -1;
    }
    cv::Mat m2 = cv::imread(imagepath2, 1);
    if (m2.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath2);
        return -1;
    }

    // Convert Image of CV Format to NCNN format
    ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels(m1.data, ncnn::Mat::PIXEL_BGR, m1.cols, m1.rows);
    ncnn::Mat ncnn_img2 = ncnn::Mat::from_pixels(m2.data, ncnn::Mat::PIXEL_BGR, m2.cols, m2.rows);

    // Detect Face
    std::vector<FaceObject> faceobjects1;
    detect_retinaface(m1, faceobjects1);
    draw_faceobjects(m1, faceobjects1);

    std::vector<FaceObject> faceobjects2;
    detect_retinaface(m2, faceobjects2);
    draw_faceobjects(m2, faceobjects2);

    // Preprocess Image
    ncnn::Mat det1 = preprocess(ncnn_img1, faceobjects1[0]);
    ncnn::Mat det2 = preprocess(ncnn_img2, faceobjects2[0]);

    // ArcFace
    Arcface arc("models");

    // Matching Face
    std::vector<float> feature1 = arc.getFeature(det1);
    std::vector<float> feature2 = arc.getFeature(det2);
    printf("Similarity: %.2f", calcSimilar(feature1, feature2));

    cv::imshow("det1", ncnn2cv(det1));
    cv::imshow("det2", ncnn2cv(det2));

    cv::waitKey(0);

    return 0;
}
