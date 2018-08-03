#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*void compute1( InputArrayOfArrays _images,
                         CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                         OutputArrayOfArrays _descriptors )
{
    if( !_descriptors.needed() )
        return;

    vector<Mat> images;

    _images.getMatVector(images);
    size_t i, nimages = images.size();

    CV_Assert( keypoints.size() == nimages );
    CV_Assert( _descriptors.kind() == _InputArray::STD_VECTOR_MAT );

    vector<Mat>& descriptors = *(vector<Mat>*)_descriptors.getObj();
    descriptors.resize(nimages);

    for( i = 0; i < 1000; i++ )
    {
        //compute1(images[i], keypoints[i], descriptors[i]);
    }
    std::cout << "1000" << endl;
}*/

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }
    
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    std::vector<KeyPoint> kepoints_1, kepoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    //Ptr<FeatureDetector> detector = BRISK::create();
    //Ptr<DescriptorExtractor> descriptor = BRISK::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
    //ORB is a binary descriptor, so can't use FlannBased directly.
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    
    detector->detect(img_1, kepoints_1);
    detector->detect(img_2, kepoints_2);
    
    //compute1(img_1, kepoints_1, descriptors_1);
    descriptor->compute(img_1, kepoints_1, descriptors_1);
    descriptor->compute(img_2, kepoints_2, descriptors_2);
    
    cout << "find " << kepoints_1.size() << "kepoints" << endl;
    
    Mat outimg1;
    drawKeypoints(img_1, kepoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("KeyPoint", outimg1);
    
    
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    
    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    
    printf("-- max_dist = %f \n", max_dist);
    printf("-- min_dist = %f \n", min_dist);
    
    vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(matches[i].distance <= max(2.0 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    
    Mat img_match, img_goodmatch;
    drawMatches(img_1, kepoints_1, img_2, kepoints_2, matches, img_match);
    drawMatches(img_1, kepoints_1, img_2, kepoints_2, good_matches, img_goodmatch);
    imshow("match", img_match);
    imshow("good_matches", img_goodmatch);
    waitKey(0);
    
    return 0;
}