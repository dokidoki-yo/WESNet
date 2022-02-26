#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <string>

using namespace std;

cv::Mat generate_fisheye_image(cv::Mat image, cv::Mat K, cv::Mat D, cv::Mat R_t_s);

int main(int argc, char **argv) {

    Eigen::Matrix3d K_F,K_L,K_B,K_R;

    //Intrinsics
    K_B<<4.2315252270666946e+02, 0., 6.3518368429424913e+02,
        0., 4.2176162080058998e+02, 5.4604808802459536e+02,
        0., 0., 1.;
    K_F<<4.2150534803053478e+02, 0., 6.2939810031193633e+02,
        0., 4.1999206255343978e+02, 5.3141472710260518e+02,
        0., 0., 1.;
    K_L<<4.2086261221668570e+02, 0., 6.4086939039393337e+02, 
        0., 4.1949874063802940e+02, 5.3582096051915732e+02,
        0., 0., 1.;
    K_R<<4.1961460580570463e+02, 0., 6.3432006841655129e+02,
        0., 4.1850638109014426e+02, 5.3932313431747673e+02,
        0., 0., 1.;
            
    //Distortion
    Eigen::Vector4d D_F,D_L,D_B,D_R;
    D_B<<-6.8507324567971206e-02, 3.3128278165505034e-03, -3.8744468086803550e-03, 7.3376684970524460e-04;
    D_F<<-6.6585685927759056e-02, -4.8144285824610098e-04, -1.1930897697190990e-03, 1.6236147741932646e-04;
    D_L<<-6.5445414949742764e-02, -6.4817440226779821e-03, 4.6429370436962608e-03, -1.4763681169119418e-03;
    D_R<<-6.6993385910155065e-02, -5.1739781929103605e-03, 7.8595773802962888e-03, -4.2367990313813440e-03;
    
    cv::Mat KB,KF,KL,KR,DB,DF,DL,DR;
    cv::eigen2cv(K_B,KB);
    cv::eigen2cv(K_F,KF);
    cv::eigen2cv(K_L,KL);
    cv::eigen2cv(K_R,KR);
    cv::eigen2cv(D_B,DB);
    cv::eigen2cv(D_F,DF);
    cv::eigen2cv(D_L,DL);
    cv::eigen2cv(D_R,DR);
    
    vector<cv::Mat> R_t_s(9);
    vector<cv::Mat> K_s(4);
    vector<cv::Mat> D_s(4);
    K_s[0]=KB;
    K_s[1]=KF;
    K_s[2]=KL;
    K_s[3]=KR;
    D_s[0]=DB;
    D_s[1]=DF;
    D_s[2]=DL;
    D_s[3]=DR;
    
    // The range of disturbance on R and t
    double ts[9]={-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2};
    double theta_s[9]={0.15,0.2,0.25,0.3,0,-0.3,-0.25,-0.2,-0.15};
    
    for(int i=0;i<9;i++){
        double theta1=theta_s[i]*10./180.*CV_PI;
        double t1=ts[i];
        Eigen::Matrix3d r_x,r_y,r_z,r,t;
        r_x << cos(theta1),-sin(theta1),0,sin(theta1),cos(theta1),0,0,0,1;
        r_y <<1,0,0,0,cos(theta1),-sin(theta1),0,sin(theta1),cos(theta1);
        r_z <<cos(theta1),0,sin(theta1),0,1,0,-sin(theta1),0,cos(theta1);
        t <<0,0,t1,0,0,t1,0,0,t1;
        r = r_x*r_y*r_z-t;
        cv::eigen2cv(r,R_t_s[i]);
    }
//     Read Images
    string root="YOUR ROOT";
    string Video_names[12]={"The name of 12 sub-dataset"};
    string camera_folder[4] = {"B","F","L","R"};
    int train_count[4] = {0,0,0,0};
    for(int i=0;i<12;i++){
            for (int j=0;j<4;j++){
            // read each image
            string original_image_name = root+Video_names[i]+"/"+camera_folder[j]+"/*.jpg";
            vector<cv::String> image_list;
            cv::glob(original_image_name,image_list);
            for (int k=0;k<image_list.size();k++){
            string image_name=image_list[k];
            cv::Mat image=cv::imread(image_name,1);
            
            //add disturbance
            for(int index = 0;index<9;index++){
                cv::Mat d_image = generate_fisheye_image(image,K_s[j],D_s[j],R_t_s[index]);
                string new_img_name;
                // for testing data
                if(k<image_list.size()*0.1){
                new_img_name="TESTING_ROOT/V"+to_string(i)+"/"+camera_folder[j]+"/"+to_string(index)+string(7-to_string(k).length(),'0')+to_string(k)+".jpg";
                cv::imwrite(new_img_name,d_image);
                cout<<"V"+to_string(i)+"/"+camera_folder[j]+"/"+to_string(index)+string(7-to_string(k).length(),'0')+to_string(k)+".jpg written done!"<<endl;
                }
                // for training data
                else{
                int count = train_count[j];
                new_img_name="TRAINING_ROOT/"+camera_folder[j]+"/"+to_string(index)+string(7-to_string(count).length(),'0')+to_string(count)+".jpg";
                cv::imwrite(new_img_name,d_image);
                cout<<camera_folder[j]+"/"+to_string(index)+string(7-to_string(count).length(),'0')+to_string(count)+".jpg written done!"<<endl;
                }
            }
             if(k>=image_list.size()*0.1)train_count[j]++;
            }
        }
    }
}

cv::Mat generate_fisheye_image(cv::Mat image, cv::Mat K, cv::Mat D, cv::Mat R_t_s){
    int rows=image.rows, cols=image.cols;
    cv::Mat image_undistorted=cv::Mat::zeros(rows,cols,CV_8UC3);
    cv::Mat image_H_undistorted=cv::Mat::zeros(rows,cols,CV_8UC3);
    vector<cv::Point2f> origin_distorted;
    for(int u=0;u<rows;u++)
        for(int v=0;v<cols;v++){
        origin_distorted.push_back(cv::Point2f(v,u));
        }
    vector<cv::Point2f>origin_undistorted_points;
    //origin undistorted points in the image
    cv::fisheye::undistortPoints(origin_distorted,origin_undistorted_points,K,D,cv::noArray(),K);
    
    // add homography
    cv::Mat H=K*R_t_s*K.inv();
    cv::Mat H_points_matrix=cv::Mat::zeros(cols*rows,3,CV_64FC1);
    for(int k=0;k<origin_undistorted_points.size();k++){
        H_points_matrix.at<double>(k,0)=origin_undistorted_points[k].x;
        H_points_matrix.at<double>(k,1)=origin_undistorted_points[k].y;
        H_points_matrix.at<double>(k,2)=1;
    }
    
    cv::Mat homography_points_matrix=H.inv()*H_points_matrix.t();
    vector<cv::Point2f> homography_undistorted_points;
    vector<cv::Point2f> H_distorted_points;
    for(int u=0;u<rows;u++)
        for(int v=0;v<cols;v++){
            double x=homography_points_matrix.at<double>(0,u*cols+v)/homography_points_matrix.at<double>(2,u*cols+v);
            double y=homography_points_matrix.at<double>(1,u*cols+v)/homography_points_matrix.at<double>(2,u*cols+v);
            homography_undistorted_points.push_back(cv::Point2f(x,y));
    }
    
    //generate nomalized points
    cv::Mat K_tr=cv::Mat::zeros(3,3,CV_64FC1);
    cv::invert(K,K_tr);
    homography_points_matrix=K_tr*homography_points_matrix;
    vector<cv::Point2f>normalized_undistorted_points;
    for(int t=0;t<origin_undistorted_points.size();t++){
            cv::Point2f temp_point;
            temp_point.x=homography_points_matrix.at<double>(0,t)/homography_points_matrix.at<double>(2,t);
            temp_point.y=homography_points_matrix.at<double>(1,t)/homography_points_matrix.at<double>(2,t);
            normalized_undistorted_points.push_back(temp_point);
    }
    
    //distort points back
    cv::fisheye::distortPoints(normalized_undistorted_points,H_distorted_points,K,D,0);
    cv::Mat image_H_distorted=cv::Mat::zeros(rows,cols,CV_8UC3);
    for(int t=0;t<origin_undistorted_points.size();t++){
        if(H_distorted_points[t].y>=0&&H_distorted_points[t].x>=0&&H_distorted_points[t].y<rows&&H_distorted_points[t].x<cols){
        image_H_distorted.at<cv::Vec3b>(t/cols,t%cols)=image.at<cv::Vec3b>(H_distorted_points[t].y,H_distorted_points[t].x);
        }
    }
    cv::Mat mask=image.clone();
    mask.setTo(cv::Scalar::all(0));
    cv::circle(mask,cv::Point(cols/2,rows/2),530,cv::Scalar(255,255,255),-1,0);
    cv::Mat image_result=cv::Mat::zeros(rows,cols,CV_8UC3);
    image_H_distorted.copyTo(image_result,mask);
    return image_result;
}
