"""Train camera pose detector."""
import torch
from torch.utils.data import DataLoader
import config
import data
import util
from model import CameraPoseDetector,Loss_Pro,Loss_Pho
from math import sqrt

def generate_pro_obj(image2d_points_set):
    image2d_points_set[:, :, :, 0] = image2d_points_set[:, :, :, 0] / 1280
    image2d_points_set[:, :, :, 1] = image2d_points_set[:, :, :, 1] / 1080
    gradient = torch.ones_like(image2d_points_set)
    print('image2d_points_set')
    # print(b.shape)
    print(image2d_points_set[0,2,0])
    return image2d_points_set,gradient

def generate_loss_pro(prediction,objective, world3d_points_set,K_set,device):
    a = Loss_Pro(K_set,device)
    return a(prediction,objective, world3d_points_set)

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


def train_detector(args):
    """Train detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    detector = CameraPoseDetector(#3,16,6
        3, args.depth_factor, args.batch_size).to(device)
    if args.detector_weights:
        print("Loading weights: %s" % args.detector_weights)
        detector.load_state_dict(torch.load(args.detector_weights))
    detector.train()


    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)
    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = util.Logger(args.enable_visdom, ['train_loss'])
    torch.multiprocessing.set_sharing_strategy('file_system')
    data_loader = DataLoader(data.CameraPoseDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))

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

    D_B = torch.tensor([-6.8507324567971206e-02, 3.3128278165505034e-03,
                        -3.8744468086803550e-03, 7.3376684970524460e-04],device=device)
    D_F = torch.tensor([-6.6585685927759056e-02, -4.8144285824610098e-04,
                        -1.1930897697190990e-03, 1.6236147741932646e-04],device=device)
    D_L = torch.tensor([-6.5445414949742764e-02, -6.4817440226779821e-03,
                        4.6429370436962608e-03, -1.4763681169119418e-03], device=device)
    D_R = torch.tensor([-6.6993385910155065e-02, -5.1739781929103605e-03,
                        7.8595773802962888e-03, -4.2367990313813440e-03], device=device)
    K_set = torch.stack([K_B,K_F,K_L,K_R])
    D_set = torch.stack([D_B,D_F,D_L,D_R])
    P_G_set = prepare_p_G(device)

    for epoch_index in range(args.num_epochs):
        for iter_idx, (images, image2d_points_set, world3d_points_set,images_gray) in enumerate(data_loader):
            #combine batch_size images
            images = torch.cat(images).to(device)
            image2d_points_set = torch.stack(image2d_points_set).to(device)
            world3d_points_set = torch.stack(world3d_points_set).to(device)
            images_gray = torch.stack(images_gray).to(device)
            optimizer.zero_grad()
            prediction = detector(images)

            #######reprojection loss
            objective_pro,gradient_pro = generate_pro_obj(image2d_points_set)#bach_zise,4,3,2,1
            loss_pro,loss_org= generate_loss_pro(prediction,objective_pro, world3d_points_set,K_set,device)

            # #######photometric loss

            loss_pho = generate_loss_pho(prediction, K_set, D_set, images_gray, P_G_set, device)
            gradient_pho = torch.ones_like(loss_pho)

            gradient_b = torch.zeros_like(loss_pro)
            gradient_f = torch.zeros_like(loss_pro)
            gradient_l = torch.zeros_like(loss_pro)
            gradient_r = torch.zeros_like(loss_pro)
            gradient_b[:, 0, :, :] = gradient_pro[:, 0, :, :]
            gradient_f[:, 1, :, :] = gradient_pro[:, 1, :, :]
            gradient_l[:, 2, :, :] = gradient_pro[:, 2, :, :]
            gradient_r[:,3,:,:]=gradient_pro[:, 3, :, :]
            gradient_org = torch.ones_like(loss_org)

            loss_pro.backward(gradient_pro*10,retain_graph=True)
            loss_org.backward(gradient_org,retain_graph=True)
            loss_pho.backward(gradient_pho*5)
            optimizer.step()

            N = torch.sum(gradient_pro).item()
            Nb=torch.sum(gradient_b).item()
            Nf = torch.sum(gradient_f).item()
            Nl = torch.sum(gradient_l).item()
            Nr = torch.sum(gradient_r).item()
            train_loss_pro = torch.sum(gradient_pro * loss_pro).item() / N
            train_lossb = torch.sum(gradient_b*loss_pro).item()/(Nb)
            train_lossf = torch.sum(gradient_f * loss_pro).item() / (Nf)
            train_lossl = torch.sum(gradient_l * loss_pro).item() / (Nl)
            train_lossr = torch.sum(gradient_r * loss_pro).item() / (Nr)
            train_loss_org = torch.sum(gradient_org*loss_org).item()/(args.batch_size*4)

            print("pro_loss")
            print(sqrt(train_loss_pro))

            logger.log(epoch=epoch_index, iter=iter_idx,train_loss=sqrt(train_loss_pro))
            print(sqrt(train_lossb))
            print(sqrt(train_lossf))
            print(sqrt(train_lossl))
            print(sqrt(train_lossr))
            print('org_loss',train_loss_org)
            print('pho_loss',loss_pho.item())
            torch.cuda.empty_cache()

            if(iter_idx%1000==0):
                torch.save(detector.state_dict(),
                           '/home/chenyang/NewDisk2/Dataset/weights1/lp_iter_detector_%d.pth' % iter_idx)

        torch.save(detector.state_dict(),
                   '/home/chenyang/NewDisk2/Dataset/weights1/lp_detector_%d.pth' % epoch_index)
        torch.save(optimizer.state_dict(), '/home/chenyang/NewDisk2/Dataset/weights1/optimizer.pth')

if __name__ == '__main__':
    train_detector(config.get_parser_for_training().parse_args())
