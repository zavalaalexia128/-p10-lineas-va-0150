# Estudiante: Zavala Medina | NC: 0150
import cv2
import numpy as np

# 1. Cargar imagen y convertir a grises
img = cv2.imread('mapachin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Detección de bordes
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# --- CÓDIGO BASE ---
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
img_base = img.copy()

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(img_base, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- EXPERIMENTO A (NC: 50) ---
lines_expA = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
img_expA = img.copy()

if lines_expA is not None:
    for x1, y1, x2, y2 in lines_expA.reshape(-1, 4):
        cv2.line(img_expA, (x1, y1), (x2, y2), (255, 0, 0), 2)

# --- EXPERIMENTO B (NC: 150) ---
lines_expB = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=150, maxLineGap=20)
img_expB = img.copy()

if lines_expB is not None:
    for x1, y1, x2, y2 in lines_expB.reshape(-1, 4):
        cv2.line(img_expB, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Visualización
cv2.imshow('Base', img_base)
cv2.imshow('Exp A (Sensible - 50)', img_expA)
cv2.imshow('Exp B (Estricto - 150)', img_expB)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Estudiante: Zavala Medina | Numero de estudiante: 0150")