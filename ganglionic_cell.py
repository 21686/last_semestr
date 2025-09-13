import numpy as np
import cv2
import matplotlib.pyplot as plt


# Модель ганглиозной клетки
class GanglionicCell:
    def __init__(self, position, k_center=5, k_surround=11, isoff=False):
        self.pos = position
        self.s1 = k_center
        self.s2 = k_surround
        self.size = (k_center, k_surround)
        self.isoff = isoff

    def get_response(self, image):
        gauss_d1 = cv2.GaussianBlur(image, (self.s1, self.s1), sigmaX=0)
        gauss_d2 = cv2.GaussianBlur(image, (self.s2, self.s2), sigmaX=0)
        if self.isoff:
            laplace_response = gauss_d2 - gauss_d1
        else:
            laplace_response = gauss_d1 - gauss_d2
        v = laplace_response[self.pos[1], self.pos[0]]
        return v


# Модель простой клетки
class VerticalSimpleCell:
    def __init__(self, position, size=5):
        self.ganglionic_cells = []
        self.size = size
        d = 3
        for i in range(-(size // 2), size // 2 + 1):
            for j in range(-(size // 2), size // 2 + 1):
                isoff = True
                if i == 0:
                    isoff = False
                self.ganglionic_cells.append(
                    GanglionicCell((position[0] + i * d, position[1] + j * d), isoff=isoff)
                )

    def get_response(self, image):
        response = 0.
        for cell in self.ganglionic_cells:
            response += cell.get_response(image)
        return response


# Модель сложной клетки (сумма простых клеток)
class ComplexCell:
    def __init__(self, position, size=5):
        self.simple_cells = []
        d = 10  # Расстояние между простыми клетками
        for i in range(-(size // 2), size // 2 + 1):
            for j in range(-(size // 2), size // 2 + 1):
                # Каждая простая клетка расположена в различном пространственном положении
                self.simple_cells.append(
                    VerticalSimpleCell((position[0] + i * d, position[1] + j * d), size=size)
                )

    def get_response(self, image):
        response = 0.
        for cell in self.simple_cells:
            response += abs(cell.get_response(image))
        return response


# Функции проверки для простой клетки
def check_point_stimulus(cell):
    responses = []
    response_map = np.zeros((256, 256), dtype=np.int16)
    for i in range(0, 13):
        for j in range(0, 13):
            image = np.zeros((256, 256), dtype=np.int16)
            cv2.circle(image, center=(128 + i - 6, 128 + j - 6), radius=1, color=(255, 255, 255), thickness=-1)
            v = cell.get_response(image)
            responses.append(v + 128)
            cv2.circle(response_map, center=(i * 16 + 32, j * 16 + 32), radius=2, color=int(v), thickness=-1)
    return response_map, responses


def check_circle_stimulus(cell):
    responses = []
    for i in range(0, 30):
        image = np.zeros((256, 256), dtype=np.int16)
        cv2.circle(image, (128, 128), radius=i, color=(255, 255, 255), thickness=i * 2 + 1)
        v = cell.get_response(image)
        responses.append(v)
    return responses


def rotate_line(cell):
    responses = []
    angles = []
    for i in range(0, 360, 10):
        angle_grad = i
        angle = i / 180 * np.pi
        image = np.zeros((256, 256), dtype=np.int16)
        cv2.line(image, (128 + int(100 * np.cos(angle)), 128 + int(100 * np.sin(angle))),
                 (128 - int(100 * np.cos(angle)), 128 - int(100 * np.sin(angle))), color=(255, 255, 255),
                 thickness=3, lineType=8)
        v = cell.get_response(image)
        responses.append(v)
        angles.append(angle_grad)
    return responses, angles


if __name__ == "__main__":
    # Ганглиозная клетка
    ganglionic_cell = GanglionicCell(position=(128, 128), k_center=5, k_surround=11, isoff=False)

    # Проверка реакции ганглиозной клетки
    response_map_ganglionic, _ = check_point_stimulus(ganglionic_cell)
    plt.imshow(response_map_ganglionic, cmap='gray')
    plt.title('Response of Ganglionic Cell to Point Stimulus')
    plt.colorbar()
    plt.show()

    responses_ganglionic = check_circle_stimulus(ganglionic_cell)
    plt.plot(responses_ganglionic)
    plt.title('Response of Ganglionic Cell to Circle Stimulus')
    plt.xlabel('Radius')
    plt.ylabel('Response')
    plt.show()

    responses_ganglionic, angles_ganglionic = rotate_line(ganglionic_cell)
    plt.plot(angles_ganglionic, responses_ganglionic)
    plt.title('Response of Ganglionic Cell to Rotating Line')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Response')
    plt.show()

    # Простая клетка
    simple_cell = VerticalSimpleCell(position=(128, 128), size=5)

    responses_simple = check_point_stimulus(simple_cell)
    plt.imshow(responses_simple[0], cmap='gray')
    plt.title('Response of Simple Cell to Point Stimulus')
    plt.colorbar()
    plt.show()

    responses_circle_simple = check_circle_stimulus(simple_cell)
    plt.plot(responses_circle_simple)
    plt.title('Response of Simple Cell to Circle Stimulus')
    plt.xlabel('Radius')
    plt.ylabel('Response')
    plt.show()

    responses_line_simple, angles_line_simple = rotate_line(simple_cell)
    plt.plot(angles_line_simple, responses_line_simple)
    plt.title('Response of Simple Cell to Rotating Line')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Response')
    plt.show()

    # Сложная клетка
    complex_cell = ComplexCell(position=(128, 128), size=5)

    responses_complex = check_point_stimulus(complex_cell)
    plt.imshow(responses_complex[0], cmap='gray')
    plt.title('Response of Complex Cell to Point Stimulus')
    plt.colorbar()
    plt.show()

    responses_circle_complex = check_circle_stimulus(complex_cell)
    plt.plot(responses_circle_complex)
    plt.title('Response of Complex Cell to Circle Stimulus')
    plt.xlabel('Radius')
    plt.ylabel('Response')
    plt.show()

    responses_line_complex, angles_line_complex = rotate_line(complex_cell)
    plt.plot(angles_line_complex, responses_line_complex)
    plt.title('Response of Complex Cell to Rotating Line')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Response')
    plt.show()
