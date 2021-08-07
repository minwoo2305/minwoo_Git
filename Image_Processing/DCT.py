import math
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg

test_img = np.array(pilimg.open('test_img.jpg'))    # 28 x 28
lenna = np.array(pilimg.open('lenna.jpg'))  # 512 x 512
image = np.array(pilimg.open('img1024x1024.bmp'))   # 1024 x 1024


class dct_and_quantization():
    def __init__(self, img, size):
        self.img = img
        self.size = size

        # 테스트 이미지

        plt.imshow(img, cmap='gray')
        plt.show()

        # A 변환행렬 size x size
        self.A = self.A_matrix()
        plt.imshow(self.A, cmap='gray')
        plt.show()

        # 결과값 저장 행렬
        self.X = np.zeros((len(img), len(img)))
        self.inverse_x = np.zeros((len(img), len(img)))

    def A_matrix(self):
        A = np.zeros((self.size, self.size))

        for n in range(self.size):
            for m in range(self.size):
                if n == 0:
                    A[n, m] = math.sqrt(1/self.size) * math.cos((((2*m)+1)*n*math.pi)/(2*self.size))
                else:
                    A[n, m] = math.sqrt(2/self.size) * math.cos((((2*m)+1)*n*math.pi)/(2*self.size))

        return A

    def fdct(self):
        # FDCT
        for i in range(0, len(self.img), self.size):
            for j in range(0, len(self.img), self.size):
                self.X[i:i + self.size, j:j + self.size] = \
                    np.dot(np.dot(self.A, self.img[i:i + self.size, j:j + self.size]), self.A.T)

        return self.X

    def idct(self):
        # IDCT
        for i in range(0, len(self.img), self.size):
            for j in range(0, len(self.img), self.size):
                self.inverse_x[i:i + self.size, j:j + self.size] = \
                    np.dot(np.dot(self.A.T, self.X[i:i + self.size, j:j + self.size]), self.A)

        return self.inverse_x

    def quantization(self, num):
        if num == 1:
            # Luminance Quantization Table
            mat = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])
        elif num == 2:
            # Chrominance Quantization Table
            mat = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                            [18, 21, 26, 66, 99, 99, 99, 99],
                            [24, 26, 56, 99, 99, 99, 99, 99],
                            [47, 66, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99]])

        else:
            mat = np.array([[8, 8, 8, 8, 8, 8, 8, 8],
                           [12, 10, 6, 3, -3, -6, -10, -12],
                           [8, 4, -4, -8, -8, -4, 4, 8],
                           [10, -3, -12, -6, 6, 12, 3, -10],
                           [8, -8, -8, 8, 8, -8, -8, 8],
                           [6, -12, 3, 10, 10, -3, 12, -6],
                           [4, -8, 8, -4, -4, 8, -8, 4],
                           [3, -6, 10, -12, 12, -10, 6, -3]])

        for i in range(0, len(self.img), self.size):
            for j in range(0, len(self.img), self.size):
                self.X[i:i + self.size, j:j + self.size] = \
                    np.rint(np.divide(self.X[i:i + self.size, j:j + self.size], mat))

        return self.X

    def Qstep_quantization(self, QP):
        Qstep = [0.625, 0.6875, 0.8125, 0.875, 1, 1.125, 1.25, 1.375, 1.625,
                 1.75, 2, 2.25, 2.5, 5, 10, 20, 40, 80, 160, 224]

        for i in range(0, len(self.img), self.size):
            for j in range(0, len(self.img), self.size):
                self.X[i:i + self.size, j:j + self.size] = \
                    np.rint(self.X[i:i + self.size, j:j + self.size] / Qstep[QP])

        return self.X


dct = dct_and_quantization(lenna, 8)    # img, matrix size
dct.fdct()
plt.imshow(dct.fdct(), cmap='gray')
plt.show()
# 1 : 8x8 Luminance / 2 : 8x8 Chrominance / 3 : 8x8 Luma-Chroma
dct.quantization(2)
dct.idct()
plt.imshow(dct.idct(), cmap='gray')
plt.show()

# Qstep Quantization
'''
for i in range(20):
    dct = dct_and_quantization(lenna, 8)  # img, matrix size
    dct.fdct()
    dct.Qstep_quantization(i)
    dct.idct()
    plt.imshow(dct.idct(), cmap='gray')
    plt.show()
'''