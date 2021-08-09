"""Inference with camera pose detector."""

import torch
import config
from model import CameraPoseDetector,Loss_Pro,Loss_Pho
from math import sqrt
import cv2 as cv
from torchvision.transforms import ToTensor
import json

INF_SIZE_IMAGES=10
def generate_obj1(image2d_points_set):
    image2d_points_set[:, :, :, 0] = image2d_points_set[:, :, :, 0] / 1280
    image2d_points_set[:, :, :, 1] = image2d_points_set[:, :, :, 1] / 1080
    gradient = torch.ones_like(image2d_points_set)
    return image2d_points_set,gradient

def generate_pro_loss(prediction,objective, world3d_points_set,K_set,device):
    a = Loss_Pro(K_set, device)
    return a(prediction, objective, world3d_points_set)

def generate_loss_pho(prediction,K_set,D_set,images_gray,P_G_set,device):
    b = Loss_Pho(K_set,D_set,P_G_set,device)
    return b(prediction,images_gray)

def prepare_p_G(device):
    ROI = torch.tensor([[300, 400, 800, 950, 600, 700, 800, 950],  # B2,3
                             [250, 350, 100, 250, 600, 700, 100, 250]], device=device)  # F2,3
    P_G_all = torch.ones([4,4, 151 * 101], device=device)
    count = 0
    for index1 in range(2):
        for index2 in range(2,4):
            x_min = ROI[index1][(index2 % 2) * 4]
            x_max = ROI[index1][(index2 % 2) * 4 + 1]
            y_min = ROI[index1][(index2 % 2) * 4 + 2]
            y_max = ROI[index1][(index2 % 2) * 4 + 3]
            nDx = 0.01
            nDy = 0.01
            nCols = 1000
            nRows = 1000
            rows = y_max - y_min+1
            cols = x_max - x_min+1
            K_G = torch.zeros([3, 3], device=device)
            K_G[0, 0] = 1 / nDx
            K_G[1, 1] = -1 / nDy
            K_G[0, 2] = nCols / 2
            K_G[1, 2] = nRows / 2
            K_G[2, 2] = 1.0
            p_G = torch.ones([3, rows * cols], device=device)
            for i in range(rows):
                for j in range(cols):
                    p_G[0, cols * i + j] = j + x_min
                    p_G[1, cols * i + j] = i + y_min

            P_G = torch.ones([4, rows * cols], device=device)
            P_G[0:3] = torch.matmul(torch.inverse(K_G), p_G)
            P_G[2, 0:rows * cols] = 0.#world coordinate:z = 0
            P_G_all[count] = P_G
            count = count+1
    return P_G_all

def prepare_inference_dataset(image_name,index):
    images = []
    images_gray = []
    image2d_points_set = []
    world3d_points_set = []
    for iter_index in range(4):
        if iter_index == 0:
            mark = 'B'
        if iter_index == 1:
            mark = 'F'
        if iter_index == 2:
            mark = 'L'
        if iter_index == 3:
            mark = 'R'
        image2d_points = []
        world3d_points = []
        image = cv.imread('/home/chenyang/NewDisk2/Dataset/testing_data/V'+str(index)+'/' + mark + '/' + image_name + '.jpg')
        image_clone = image.copy()
        image_gray = cv.cvtColor(image_clone, cv.COLOR_BGR2GRAY)
        transform = ToTensor()
        images_gray.append(transform(image_gray))
        image = cv.resize(image, (512, 512))
        image = transform(image)
        images.append(image)
        with open('/home/chenyang/NewDisk2/Dataset/training_data/C_f_labels/label'+ image_name[0] + '_'+mark+'.json') as file:
            for point in json.load(file)['world3d_points']:
                world3d_points.append(point)

        with open('/home/chenyang/NewDisk2/Dataset/training_data/C_f_labels/label' + image_name[0] + '_' + mark + '.json') as file:
            for point in json.load(file)['image2d_points']:
                image2d_points.append(point)
        image2d_points = torch.Tensor(image2d_points)
        world3d_points = torch.Tensor(world3d_points)
        image2d_points_set.append(image2d_points)
        world3d_points_set.append(world3d_points)
    all_images = torch.stack([images[0], images[1], images[2], images[3]])
    all_gray_images = torch.stack([images_gray[0], images_gray[1], images_gray[2], images_gray[3]])
    all_2d_points = torch.stack(
        [image2d_points_set[0], image2d_points_set[1], image2d_points_set[2], image2d_points_set[3]])
    all_3d_points = torch.stack(
        [world3d_points_set[0], world3d_points_set[1], world3d_points_set[2], world3d_points_set[3]])
    return all_images, all_2d_points, all_3d_points, all_gray_images

def train_detector(args):
    """Train detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)
    detector = CameraPoseDetector(#3,16,6
        3, args.depth_factor, 1).to(device)
    detector.load_state_dict(torch.load(args.detector_weights))
    detector.eval()

    D_B = torch.tensor([-6.8507324567971206e-02, 3.3128278165505034e-03,
                        -3.8744468086803550e-03, 7.3376684970524460e-04], device=device)
    D_F = torch.tensor([-6.6585685927759056e-02, -4.8144285824610098e-04,
                        -1.1930897697190990e-03, 1.6236147741932646e-04], device=device)
    D_L = torch.tensor([-6.5445414949742764e-02, -6.4817440226779821e-03,
                        4.6429370436962608e-03, -1.4763681169119418e-03], device=device)
    D_R = torch.tensor([-6.6993385910155065e-02, -5.1739781929103605e-03,
                        7.8595773802962888e-03, -4.2367990313813440e-03], device=device)
    D_set = torch.stack([D_B, D_F, D_L, D_R])
    K_B = torch.tensor([[4.2315252270666946e+02, 0., 6.3518368429424913e+02],
                        [0., 4.2176162080058998e+02, 5.4604808802459536e+02],
                        [0., 0., 1.]], device=device)
    K_F = torch.tensor([[4.2150534803053478e+02, 0., 6.2939810031193633e+02],
                        [0., 4.1999206255343978e+02, 5.3141472710260518e+02],
                        [0., 0., 1.]], device=device)
    K_L = torch.tensor([[4.2086261221668570e+02, 0., 6.4086939039393337e+02],
                        [0., 4.1949874063802940e+02, 5.3582096051915732e+02],
                        [0., 0., 1.]], device=device)
    K_R = torch.tensor([[4.1961460580570463e+02, 0., 6.3432006841655129e+02],
                        [0., 4.1850638109014426e+02, 5.3932313431747673e+02],
                        [0., 0., 1.]], device=device)

    K_set = torch.stack([K_B,K_F,K_L,K_R])

    P_G_set = prepare_p_G(device)

    image_names = []
    for index in range(1,2):
        for image_index in range(INF_SIZE_IMAGES):
            image_names.append(str(index) + str(image_index).zfill(7))
    for folder_id in range(12):
        total_loss_b = 0.
        total_loss_f = 0.
        total_loss_l = 0.
        total_loss_r = 0.
        total_loss = 0.
        total_loss_pho = 0.
        for name_index in range(len(image_names)):
            inference_dataset, image2d_points_set, world3d_points_set,images_grays = prepare_inference_dataset(image_names[name_index],folder_id)
            prediction = detector(inference_dataset.to(device))
            objective_pro,gradient = generate_obj1(image2d_points_set.unsqueeze(0))
            inf_loss_pro,inf_loss_org = generate_pro_loss(prediction, objective_pro, world3d_points_set.unsqueeze(0), K_set, device)
            inf_loss_pho = generate_loss_pho(prediction,K_set,D_set,images_grays.unsqueeze(0).to(device),P_G_set,device)
            gradient_pho = torch.ones_like(inf_loss_pho)
            gradient_org = torch.ones_like(inf_loss_pro)
            gradient_b = torch.zeros_like(gradient)
            gradient_f = torch.zeros_like(gradient)
            gradient_l = torch.zeros_like(gradient)
            gradient_r = torch.zeros_like(gradient)
            gradient_b[:, 0, :, :] = gradient[:,0, :, :]
            gradient_f[:, 1, :, :] = gradient[ :,1, :, :]
            gradient_l[:, 2, :, :] = gradient[:, 2, :, :]
            gradient_r[:, 3, :, :] = gradient[ :,3, :, :]

            N = torch.sum(gradient).item()
            Nb = torch.sum(gradient_b).item()
            Nf = torch.sum(gradient_f).item()
            Nl = torch.sum(gradient_l).item()
            Nr = torch.sum(gradient_r).item()
            info_lossb = torch.sum(gradient_b * inf_loss_pro).item() / (Nb)
            info_lossf = torch.sum(gradient_f * inf_loss_pro).item() / (Nf)
            info_loss = torch.sum(gradient * inf_loss_pro).item() / N
            info_lossl = torch.sum(gradient_l * inf_loss_pro).item() / (Nl)
            info_lossr = torch.sum(gradient_r * inf_loss_pro).item() / (Nr)
            info_loss_org = torch.sum(inf_loss_org).item() /  4
            train_loss_pho = torch.sum(gradient_pho * inf_loss_pho).item()/  4
            total_loss_b = total_loss_b+sqrt(info_lossb)
            total_loss_f = total_loss_f + sqrt(info_lossf)
            total_loss_l = total_loss_l + sqrt(info_lossl)
            total_loss_r = total_loss_r + sqrt(info_lossr)
            total_loss = total_loss + sqrt(info_loss)
            total_loss_pho = total_loss_pho + train_loss_pho
            # print("inference", name_index, "done!")

        print('folder total done!',folder_id)
        print('b',total_loss_b/len(image_names))
        print('f',total_loss_f/len(image_names))
        print('l',total_loss_l/len(image_names))
        print('r',total_loss_r/len(image_names))
        print('average_loss_pro',total_loss/len(image_names))
        print('loss_pho',total_loss_pho/len(image_names))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train_detector(config.get_parser_for_inference().parse_args())
