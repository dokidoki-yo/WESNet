import torch
import torch.nn as nn
import cv2
from numpy import *
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F


class Loss_Pho(nn.modules.Module):
    def __init__(self, K_set,D_set,P_G_set,device):
        super(Loss_Pho, self).__init__()
        self.device = device
        self.K_set = K_set
        self.D_set = D_set
        self.P_G_set = P_G_set
        self.ROI = torch.tensor([[300, 400, 800, 950, 600, 700, 800, 950],  # B2,3
                             [250, 350, 100, 250, 600, 700, 100, 250]], device=device)  # F2,3

    def forward(self, prediction,images_gray):
        loss_pho = torch.tensor(0., device=self.device)
        for batch_index in range(len(prediction)):
            T_B, T_F, T_L, T_R= self.generate_T_matrix(prediction, batch_index)
            gradientL = self.generate_gradient(images_gray[batch_index, 2,0])
            gradientR = self.generate_gradient(images_gray[batch_index, 3,0])
            loss_BL,num_BL = self.generate_single_loss(T_B, T_L, images_gray[batch_index], 0, 2,0,gradientL)
            loss_BR,num_BR= self.generate_single_loss(T_B, T_R, images_gray[batch_index], 0, 3,1,gradientR)
            loss_FL, num_FL= self.generate_single_loss(T_F, T_L, images_gray[batch_index], 1, 2,2,gradientL)
            loss_FR, num_FR= self.generate_single_loss(T_F, T_R, images_gray[batch_index], 1, 3,3,gradientR)
            print(loss_FR)
            loss_pho = loss_pho + loss_BL + loss_BR + loss_FL + loss_FR
        return loss_pho

    def generate_T_matrix(self,prediction,batch_index):
        all_matrix = torch.zeros([4,4,4], device=self.device)
        for index in range(4):
            transform_matrix = torch.zeros([4, 4], device=self.device)
            rotation_matrix = prediction[batch_index, index * 9:index * 9 + 9].reshape((3, 3))
            transform_matrix[0:3, 0:3] = rotation_matrix
            transform_matrix[0, 3] = prediction[batch_index, 36 + index * 3]
            transform_matrix[1, 3] = prediction[batch_index, 37 + index * 3]
            transform_matrix[2, 3] = prediction[batch_index, 38 + index * 3]
            transform_matrix[3, 3] = 1.
            all_matrix[index] = transform_matrix
        return all_matrix[0],all_matrix[1],all_matrix[2],all_matrix[3]


    def nn_conv2d(self,im):
        # 用nn.Conv2d定义卷积操作
        conv_op = nn.Conv2d(1, 1, 3, bias=False)
        # 定义sobel算子参数
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        # 将sobel算子转换为适配卷积操作的卷积核
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        # 给卷积操作的卷积核赋值
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        # 对图像进行卷积操作
        edge_detect = conv_op(Variable(im))
        # 将输出转换为图片格式
        edge_detect = edge_detect.squeeze().detach().numpy()
        return edge_detect

    def functional_conv2d(self,im):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') #
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = Variable(torch.from_numpy(sobel_kernel).to(self.device))
        edge_detect = F.conv2d(Variable(im), weight,padding=1)
        edge_detect = edge_detect.squeeze()
        return edge_detect

    def generate_single_loss(self,camera_pose1,camera_pose2,gray,index1,index2,P_G_index,gradient):
        undistort1 = self.generate_table(camera_pose1,index1, self.P_G_set[P_G_index])
        undistort2 = self.generate_table(camera_pose2,index2, self.P_G_set[P_G_index])

        x_min = self.ROI[index1][(index2%2)*4]
        x_max = self.ROI[index1][(index2%2)*4+1]
        y_min = self.ROI[index1][(index2%2)*4+2]
        y_max = self.ROI[index1][(index2%2)*4+3]

        gray_pho1 = torch.zeros((y_max - y_min + 1)*(x_max - x_min + 1), device=self.device)
        gray_pho2 = torch.zeros_like(gray_pho1)
        count = 0
        mean = torch.sum(gradient[y_min:y_max,x_min:x_max])/((y_max-y_min)*(x_max-x_min))
        std = torch.std(gradient[y_min:y_max,x_min:x_max])
        for undistorted_x in range(x_min,x_max):
            for undistorted_y in range(y_min,y_max):
                off_set_x = undistorted_x - x_min
                off_set_y = undistorted_y - y_min
                if gradient[undistorted_y,undistorted_x]>2*mean+std:
                    a = self.generate_pixel(off_set_y,off_set_x,undistort1,gray[index1,0],self.K_set[index1],self.D_set[index1])
                    b = self.generate_pixel(off_set_y,off_set_x,undistort2, gray[index2, 0],self.K_set[index2],self.D_set[index2])
                    gray_pho1[count]=a
                    gray_pho2[count]=b
                    count = count+1
        del undistort1,undistort2
        gray_pho1_cut = gray_pho1[0:count]
        gray_pho2_cut = gray_pho2[0:count]
        mean1 = torch.sum(gray_pho1_cut)
        mean2 = torch.sum(gray_pho2_cut)
        coeff = 0
        if mean2!=0:
            coeff = mean1/mean2
        loss = torch.nn.L1Loss()

        photo_loss = loss(gray_pho1_cut,coeff*gray_pho2_cut)
        print('count', count, 'loss',photo_loss)
        return photo_loss,count

    def generate_pixel(self,offset_y,offset_x, un_point,img,K,D):
        x = un_point[0,offset_y,offset_x]
        y = un_point[1,offset_y,offset_x]
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1
        den = (x2 - x1) * (y2 - y1)
        w11 = (x2 - x) * (y2 - y) / den
        w21 = (x - x1) * (y2 - y) / den
        w12 = (x2 - x) * (y - y1) / den
        w22 = (x - x1) * (y - y1) / den
        h_points = torch.tensor([[x1,y1,1],
                                 [x2,y1,1],
                                 [x1,y2,1],
                                 [x2,y2,1]],device=self.device)
        norm_points = torch.mm(K.inverse().double(),h_points.t())
        norm_points = norm_points.t().cpu().detach().numpy()
        norm_points_cv = np.dstack([norm_points[:,0],norm_points[:,1]])
        K = K.cpu().detach().numpy()
        D = D.cpu().detach().numpy()
        table = cv2.fisheye.distortPoints(norm_points_cv, K, D)[0]
        img11 = img[table[0, 1].astype(int), table[0, 0].astype(int)]
        img21 = img[table[1,1].astype(int), table[1,0].astype(int)]
        img12 = img[table[2, 1].astype(int), table[2, 0].astype(int)]
        img22 = img[table[3, 1].astype(int), table[3, 0].astype(int)]
        img = w11*img11+w21*img21+w12*img12+w22*img22
        return img

    def bilinear_intp(self,x,y,img):
        x1 = np.floor(x).astype(int)
        x2 = x1+1
        y1 = np.floor(y).astype(int)
        y2 = y1+1
        den = (x2-x1)*(y2-y1)
        w11 = (x2-x)*(y2-y)/den
        w21 = (x-x1)*(y2-y)/den
        w12 = (x2-x)*(y-y1)/den
        w22 = (x-x1)*(y-y1)/den
        pixel= w11*img[y1,x1]+w21*img[y1,x2]+w12*img[y2,x1]+w22*img[y2,x2]
        return pixel

    def generate_table(self,camera_pose,index1,P_G):
        cols = 100+1
        rows = 150+1
        K = self.K_set[index1]
        P_GC = torch.mm(camera_pose, P_G)
        channel_x = P_GC[0].double()/P_GC[2].double()
        channel_y = P_GC[1].double()/P_GC[2].double()
        channel_z = P_GC[2].double()/P_GC[2].double()
        undistorted_points_norm = torch.stack([channel_x,channel_y,channel_z])
        undistorted_points = torch.mm(K.double(),undistorted_points_norm)
        return undistorted_points.reshape(3,rows,cols)

    def generate_gradient(self,im):
        im = im.reshape((1, 1, im.shape[0], im.shape[1]))
        edge_detect = self.functional_conv2d(im)
        return  torch.abs(edge_detect)


