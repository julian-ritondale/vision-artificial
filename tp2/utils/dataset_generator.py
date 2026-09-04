import cv2
import csv
import glob
import numpy
import math

try:
    from utils.label_converters import label_to_int
except ModuleNotFoundError:
    from label_converters import label_to_int

BLOCK_SIZE = 45
C_CONSTANT = 10

def generate_hu_moments_file():
    with open('dataset/tetris-hu-moments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        write_hu_moments("cleveland_z", writer)
        write_hu_moments("hero", writer)
        write_hu_moments("orange_ricky", writer)
        write_hu_moments("smashboy", writer)
        write_hu_moments("teewee", writer)

def write_hu_moments(label, writer):
    files = glob.glob('./tetris/' + label + '/*')
    for file in files:
        hu = hu_moments_of_file(file)
        if hu is None:
            continue
        row = numpy.append(hu.ravel(), label_to_int(label))
        writer.writerow(row)

def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, BLOCK_SIZE, C_CONSTANT
    )
    bin_img = 255 - bin_img

    kernel = numpy.ones((3, 3), numpy.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  No contour found: {filename}")
        return None
    shape_contour = max(contours, key=cv2.contourArea)

    filled = numpy.zeros(bin_img.shape, dtype=numpy.uint8)
    cv2.fillPoly(filled, [shape_contour], 255)

    preview = image.copy()
    cv2.drawContours(preview, [shape_contour], -1, (0, 255, 0), 2)

    cv2.namedWindow("Contour Preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Contour Preview", preview)
    cv2.imshow("Binary Mask", bin_img)

    print(f"  {filename}  —  ENTER = accept   ESC/Q = skip")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):   # Enter or Space
            break
        elif key in (27, ord('q')):
            print(f"  Skipped.")
            return None

    moments = cv2.moments(filled)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(7):
        value = float(hu_moments[i])
        magnitude = abs(value)
        if magnitude == 0.0:
            hu_moments[i] = 0.0
        else:
            hu_moments[i] = -1 * math.copysign(1.0, value) * math.log10(magnitude)
    return hu_moments

if __name__ == "__main__":
    cv2.namedWindow("Contour Preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
    generate_hu_moments_file()
    cv2.destroyAllWindows()
    print("\nDataset generation complete.")