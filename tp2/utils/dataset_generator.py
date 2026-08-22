import cv2
import csv
import glob
import numpy
import math

from utils.label_converters import label_to_int

def generate_hu_moments_file():
    with open('dataset/tetris-hu-moments.csv', 'w',
              newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. 
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("cleveland_z", writer)
        write_hu_moments("hero", writer)
        write_hu_moments("orange_ricky", writer)
        write_hu_moments("smashboy", writer)
        write_hu_moments("teewee", writer)

# Escribo los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer):
    files = glob.glob('./tetris/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = numpy.append(flattened, label_to_int(label))  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.

def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima

    # Descomentar para chequear que estemos agarrando bien el contorno
    # cv2.drawContours(image, [shape_contour], -1, (0, 255, 0), 2)
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("test GRAY", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", image)
    # cv2.imshow("test GRAY", gray)
    # cv2.waitKey(0)

    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments).flatten()  # momentos de Hu, a 1D para operar por escalar
    # Log scale hu moments
    for i in range(0, 7):
        value = float(huMoments[i])
        magnitude = abs(value)
        if magnitude == 0.0:
            huMoments[i] = 0.0
        else:
            huMoments[i] = -1 * math.copysign(1.0, value) * math.log10(magnitude)  # Mapeo para agrandar la escala.
    return huMoments
