import torch
import torch.nn as nn

class Loss_Pro(nn.modules.Module):
    """Loss for Re-Projection Loss."""
    def __init__(self, K_set,device):
        super(Loss_Pro, self).__init__()
        self.K_set = K_set
        self.device = device

    def forward(self, prediction,objective, world3d_points_set):
        result = torch.zeros_like(objective)
        det_loss = torch.tensor(0., device=self.device)
        org_loss = torch.tensor(0., device=self.device)
        det_obj = torch.tensor(1., device=self.device)
        for batch_index in range(len(prediction)):
            for index in range(4):
                K = self.K_set[index]
                transform_matrix = torch.zeros([4,4],device=self.device)
                rotation_matrix = prediction[batch_index, index*9:index*9+9].reshape((3,3))
                transform_matrix[0:3,0:3]=rotation_matrix
                transform_matrix[0,3] = prediction[batch_index,36+index*3]
                transform_matrix[1, 3] = prediction[batch_index,37+index*3]
                transform_matrix[2, 3] = prediction[batch_index,38+index*3]
                transform_matrix[3,3]=1.
                world3d_points_batch = world3d_points_set[batch_index, index]
                num_points = len(world3d_points_batch)
                for point_index in range(num_points):
                    H_world_point = torch.tensor([[world3d_points_batch[point_index][0]],
                                                  [world3d_points_batch[point_index][1]],
                                                  [world3d_points_batch[point_index][2]],
                                                  [1.]], device=self.device)
                    camera_point = torch.mm(transform_matrix, H_world_point)
                    norm = camera_point[2, :]+torch.tensor(0.00001,device=self.device)
                    camera_h_point = camera_point.double() / norm.double()
                    image2d_point_H = torch.mm(K.double(), camera_h_point[0:3,:].double())
                    image2d_point_predicted = image2d_point_H[0:2, :]
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
        return (result-objective)**2, org_loss+det_loss