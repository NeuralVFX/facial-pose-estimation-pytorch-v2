import random
import json
import numpy as np
import cv2
import torch


############################################################################
#  Loader Utilities
############################################################################


def random_offset(z):
    # Random value for depth in scene
    return ((random.random() * 2) - 1) * ((z + .3) / 4)


def c_rand(mult):
    # Random value for rotation
    return ((random.random() * 2) - 1) * mult


def draw_line(image, points, color):
    # Draw line across several points
    line = np.zeros(image.shape, dtype=np.uint8)
    cv2.polylines(line, np.int32([points]), False, color, 1, cv2.LINE_AA, )
    line = np.array(line, dtype='float') / 255.
    cv2.max(image, line, image)


class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img):
        # normalize image to feed to network
        return (img - self.mean) / self.std

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data
        img = img.numpy().transpose(1, 2, 0)
        return img * self.std + self.mean


############################################################################
#  Data Generator
############################################################################


class LandMarkGenerator:
    # Loads 3d face points and blendshapes, create random mixes of blendshapes, and renders with OpenCV
    def __init__(self, transform, output_res=96, size=3000, blendshapes='./data/bs_points_a.json', rand_rot=.7):

        with open(blendshapes) as json_file:
            self.data = json.load(json_file)

        self.rand_rot = rand_rot
        self.size = size
        self.color_list = [(0, 0, 0), (0, 0, 128), (128, 0, 0), (128, 128, 256), (0, 256, 256),
                           (256, 0, 256), (256, 256, 128), (128, 0, 256), (128, 128, 128),
                           (128, 0, 128), (128, 256, 0), (256, 256, 256), (256, 128, 0),
                           (0, 128, 128), (0, 0, 256), (256, 128, 128), (0, 256, 128),
                           (128, 128, 0), (256, 128, 256), (0, 128, 256), (128, 256, 256),
                           (256, 0, 0), (256, 256, 0), (0, 128, 0), (0, 256, 0), (256, 0, 128),
                           (128, 256, 128)]

        self.transform = transform
        self.res = output_res

        # Setup camera parameters
        size = (1080, 1920, 3)
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        self.dist_coeffs = np.zeros((4, 1))

        self.face = np.array([self.data['default'][k] for k in self.data['default'].keys()])
        self.bs_list = ['BS.Mesh'] + [f'BS.Mesh{num}' for num in range(1, 51)]

    def get_pred_face(self, mult_list):
        # Create 3d points from blendshape values, render to OpenCV image, no translation randomization
        face = np.copy(self.face)

        count = 0
        for key in self.bs_list:
            bs = np.array([self.data['blend_shapes'][key][k] for k in self.data['default'].keys()])
            mult = mult_list[0, count]
            face = face + (bs * mult)
            count += 1
        (face2d, jacobian) = cv2.projectPoints(face, (0, 0, 0), (0, 0, -.3), self.camera_matrix, self.dist_coeffs)
        return self.draw(face2d, self.res)

    def get_random_blend(self):
        # Create 3d points using random blendshape values
        face = np.copy(self.face)
        rand_mult_list = []

        count = 0
        # Turned off a couple blend shapes which never performed well during inference
        ok_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 23, 24, 31, 32, 33, 34, 38, 39, 40,
                  42, 43, 44, 45, 46, 47, 50]
        for key in self.bs_list:
            bs = np.array([self.data['blend_shapes'][key][k] for k in self.data['default'].keys()])
            rand_mult = (random.random() ** 4) * (count in ok_ids)
            face = face + (bs * rand_mult)
            rand_mult_list.append(rand_mult)
            count += 1
        return face, np.array(rand_mult_list)

    def project(self, face):
        # Convert 3d points into 2d points
        z = (-3 * (random.random() ** 2)) - .3
        (face2d, jacobian) = cv2.projectPoints(face,
                                               (c_rand(self.rand_rot), c_rand(self.rand_rot), c_rand(self.rand_rot)),
                                               (random_offset(z), random_offset(z), z),
                                               self.camera_matrix
                                               , self.dist_coeffs)
        return face2d

    def draw(self, face2d, res):
        # Draw lines through 2d points
        image = np.zeros([res, res, 3])
        width = face2d[:, :, 0].max() - face2d[:, :, 0].min()
        height = face2d[:, :, 1].max() - face2d[:, :, 1].min()
        face2d = (face2d / max(width, height)) * (res * .9)

        mean_width = np.mean([face2d[:, :, 0].max(), face2d[:, :, 0].min()])
        mean_height = np.mean([face2d[:, :, 1].max(), face2d[:, :, 1].min()])

        face2d[:, :, 0] -= mean_width
        face2d[:, :, 1] -= mean_height
        face2d[:, :, :] += res // 2

        color_list = self.color_list.copy()
        # chin #
        draw_line(image, face2d[0:5], color_list.pop())
        # right eye lid #
        draw_line(image, face2d[5:10], color_list.pop())
        # left eye lid #
        draw_line(image, face2d[10:15], color_list.pop())
        # eyes #
        draw_line(image, face2d[24:28], color_list.pop())
        draw_line(image, np.concatenate((face2d[27:30], face2d[24:25]), axis=0), color_list.pop())
        draw_line(image, face2d[30:34], color_list.pop())
        draw_line(image, np.concatenate((face2d[33:36], face2d[30:31]), axis=0), color_list.pop())
        # outer mouth #
        draw_line(image, face2d[36:40], color_list.pop())
        draw_line(image, face2d[39:43], color_list.pop())
        draw_line(image, face2d[42:46], color_list.pop())
        draw_line(image, np.concatenate((face2d[45:48], face2d[36:37]), axis=0), color_list.pop())
        # inner mouth #
        draw_line(image, face2d[48:51], color_list.pop())
        draw_line(image, face2d[50:53], color_list.pop())
        draw_line(image, face2d[52:55], color_list.pop())
        draw_line(image, np.concatenate((face2d[54:56], face2d[48:49]), axis=0), color_list.pop())
        # nose #
        draw_line(image, face2d[15:19], color_list.pop())
        draw_line(image, face2d[18:20], color_list.pop())
        draw_line(image, face2d[19:22], color_list.pop())
        draw_line(image, face2d[21:24], color_list.pop())
        draw_line(image, np.concatenate((face2d[18:19], face2d[23:24]), axis=0), color_list.pop())
        return image

    def generate(self):
        # Generate X and Y training pair
        face_points_3d, blend_weights = self.get_random_blend()
        face2d = self.project(face_points_3d)
        image = self.draw(face2d, self.res)

        norm_img = np.rollaxis(self.transform.norm(image), 2)
        return norm_img, blend_weights

    def __getitem__(self, index):
        image, matrix = self.generate()
        return torch.FloatTensor(image), torch.FloatTensor(matrix)

    def __len__(self):
        return self.size
