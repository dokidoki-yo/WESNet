import torch
import torch.nn as nn

class Loss_Pro(nn.modules.Module):
    """Loss for Re-Projection Loss."""
    def __init__(self, K_B,K_F,K_L,K_R,device):
        super(Loss_Pro, self).__init__()
        self.K_B = K_B
        self.K_F = K_F
        self.K_L = K_L
        self.K_R = K_R
        self.device = device

    def forward(self, prediction,objective, world3d_points_set):
        result = torch.zeros_like(objective)
        det_loss = torch.tensor(0., device=self.device)
        org_loss = torch.tensor(0., device=self.device)
        det_obj = torch.tensor(1., device=self.device)
        for batch_index in range(len(prediction)):
            for index in range(4):
                if index == 0:
                    K = self.K_B
                if index == 1:
                    K = self.K_F
                if index == 2:
                    K = self.K_L
                if index == 3:
                    K = self.K_R
                transform_matrix = torch.zeros([4,4],device=self.device)
                rotation_matrix = prediction[batch_index, index*12:index*12+9].reshape((3,3))
                transform_matrix[0:3,0:3]=rotation_matrix
                transform_matrix[0,3] = prediction[batch_index,9+index*12]
                transform_matrix[1, 3] = prediction[batch_index,10+index*12]
                transform_matrix[2, 3] = prediction[batch_index,11+index*12]
                transform_matrix[3,3]=1.
                if batch_index==0 and index == 0:
                    print(transform_matrix)
                print(transform_matrix)
                world3d_points_batch = world3d_points_set[batch_index, index]
                num_points = len(world3d_points_batch)
                for point_index in range(num_points):
                    H_world_point = torch.tensor([[world3d_points_batch[point_index][0]],
                                                  [world3d_points_batch[point_index][1]],
                                                  [world3d_points_batch[point_index][2]],
                                                  [1.]], device=self.device)

                    camera_point = torch.mm(transform_matrix, H_world_point)
                    camera_h_point = camera_point.double() / camera_point[2, :].double()
                    image2d_point_H = torch.mm(K.double(), camera_h_point[0:3,:].double())
                    image2d_point_predicted = image2d_point_H[0:2, :]
                    #normalize
                    image2d_point_predicted[0, 0] = image2d_point_predicted[0, 0] / 1280
                    image2d_point_predicted[1, 0] = image2d_point_predicted[1, 0] / 1080
                    result[batch_index, index, point_index] = image2d_point_predicted[:,0]

                rotation_matrix_T = rotation_matrix.t()
                ones_matrix = torch.tensor([[1., 0., 0.],
                                            [0., 1., 0.], [0., 0., 1.]], device=self.device)
                tmpt = rotation_matrix
                det1 = tmpt[0, 0] * (tmpt[1, 1] * tmpt[2, 2] - tmpt[1, 2] * tmpt[2, 1])
                det2 = tmpt[1, 0] * (
                        tmpt[0, 1] * tmpt[2, 2] - tmpt[0, 2] * tmpt[2, 1])
                det3 = tmpt[2, 0] * (
                        tmpt[0, 1] * tmpt[1, 2] - tmpt[1, 1] * tmpt[0, 2])
                det = det1 - det2 + det3
                det_loss = det_loss + (det - det_obj) ** 2
                org_loss = org_loss+(torch.matmul(rotation_matrix,rotation_matrix_T)-ones_matrix)**2
        return (result-objective)**2, org_loss ,det_loss