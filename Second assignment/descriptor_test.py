
from functions import *


def main():
    # Load image as grayscale
    image = cv2.imread('im1.png', cv2.IMREAD_GRAYSCALE)

    # Parameters
    rho_min = 5
    rho_max = 20
    rho_step = 1
    N = 8

    #rotate image with angle theta1 and theta2
    p = [100, 100]
    theta1=13
    theta2=37
    #rotate the image with angles theta1 and theta2 around the point p
    image_center =p
    rot_mat = cv2.getRotationMatrix2D(image_center, theta1, 1.0)
    # rotate image
    rotated_image1 = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rot_mat = cv2.getRotationMatrix2D(image_center, theta2, 1.0)
    # rotate image
    rotated_image2 = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=255)



    # Compute descriptors for p=[100,100] and 2 different angles
    print("Basic local descriptor for im1.png with rotation angle theta1")
    descriptor1 = myLocalDescriptor(rotated_image1, p, rho_min, rho_max, rho_step, N)
    print(descriptor1)
    print("Basic local descriptor for im1.png with rotation angle theta2")
    descriptor2 = myLocalDescriptor(rotated_image2, p, rho_min, rho_max, rho_step, N)
    print(descriptor2)
    print("Upgraded local descriptor for im1.png with rotation angle theta1")
    upgraded_descriptor1 = myLocalDescriptorUpgrade(rotated_image1, p, rho_min, rho_max, rho_step, N)
    print(upgraded_descriptor1)
    print("Upgraded local descriptor for im1.png with rotation angle theta2")
    upgraded_descriptor2 = myLocalDescriptorUpgrade(rotated_image2, p, rho_min, rho_max, rho_step, N)
    print(upgraded_descriptor2)

    #Compute descriptors for q1=[200,200] and q2=[202,202] for im1.png
    q1=[200,200]
    q2=[202,202]


    print("Basic local descriptor for im1.png and point q1=[200,200]")
    descriptor1 = myLocalDescriptor(image, q1, rho_min, rho_max, rho_step, N)
    print(descriptor1)
    print("Upgraded local descriptor for im1.png and point q1=[200,200]")
    upgraded_descriptor1 = myLocalDescriptorUpgrade(image, q1, rho_min, rho_max, rho_step, N)
    print(upgraded_descriptor1)
    print("Basic local descriptor for im1.png and point q2=[202,202]")
    descriptor2 = myLocalDescriptor(image, q2, rho_min, rho_max, rho_step, N)
    print(descriptor2)
    print("Upgraded local descriptor for im1.png and point q2=[202,202]")
    upgraded_descriptor2 = myLocalDescriptorUpgrade(image, q2, rho_min, rho_max, rho_step, N)
    print(upgraded_descriptor2)