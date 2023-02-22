import math
from math import log10, floor
import numpy
import decimal
from decimal import Decimal

numpy.set_printoptions(precision=7, suppress=True, linewidth=100)

# ========== QUESTION 1 ==========

def nevilles_method(x_points, y_points, x):
    
    matrix1 = numpy.zeros((3, 3))
    
    for i in range(0, 3):
        matrix1[i][0] = y_points[i]
    
    for i in range(1, 3):
        for j in range(1, i + 1):
            first_multiplication = (x - x_points[i-j]) * matrix1[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix1[i-1][j-1]
            denominator = x_points[i] - x_points[i-1]
            
            coefficient = (first_multiplication - second_multiplication)/(x_points[i]-x_points[i-j]);
            matrix1[i][j] = coefficient
    
    q1_1 = (1/(x_points[1]-x_points[0]))*((x-x_points[0])*matrix1[1][0] - (x-x_points[1])*matrix1[0][0])
    q2_1 = (1/(x_points[2]-x_points[1]))*((x-x_points[1])*matrix1[2][0] - (x-x_points[2])*matrix1[1][0])
    q2_2 = (1/(x_points[2]-x_points[0]))*((x-x_points[0])*matrix1[2][1] - (x-x_points[2])*matrix1[1][1])
    return q2_2

x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
approximating_value = 3.7
ans1 = nevilles_method(x_points, y_points, approximating_value)

print(ans1, "\n")
    
# ========== QUESTION 2 ==========

def newton_forward(x_points, y_points, est):
    
    lim = len(x_points);
    matrix2 = numpy.zeros((lim, lim));
    
    for i in range(0, lim):
        matrix2[i][0] = y_points[i]
        
    for i in range(1, lim):
        for j in range(1, i + 1):
            matrix2[i][j] = (matrix2[i][j-1] - matrix2[i-1][j-1])/(x_points[i] - x_points[i-j])
            
    degrees = numpy.zeros(3)
    
    degrees[0] = y_points[0] + matrix2[1][1] * (est - x_points[0])
    degrees[1] = degrees[0] + matrix2[2][2] * (est - x_points[0]) * (est - x_points[1])
    degrees[2] = degrees[1] + matrix2[3][3] * (est - x_points[0]) * (est - x_points[1]) * (est - x_points[2])
    
    poly_matrix = numpy.zeros(3)
    
    for i in range(1, len(matrix2)):
        poly_matrix[i-1] = matrix2[i][i]
    
    print("[", end = '')

    for i in range (0, len(poly_matrix)):
        print(poly_matrix[i], end = '')
        if (i + 1 != len(poly_matrix)):
            print(", ", end = '');

    print("]\n")
    
    return degrees
    
x = [7.2, 7.4, 7.5, 7.6]
y = [23.5492, 25.3913, 26.8224, 27.4589]
val = 7.3

ans2 = newton_forward(x, y, val)

# ========== QUESTION 3 ==========

print(ans2[2], "\n") # Answer coded in QUESTION 2

# ========== QUESTION 4 ==========
    
def hermite_interpolation(x_points, y_points, slopes):
    
    num_of_points = len(x_points)
    matrix4 = numpy.zeros((2*len(x_points), 2*len(x_points)))
  
    for i in range(0, num_of_points):
        matrix4[2*i][0] = x_points[i]
        matrix4[2*i + 1][0] = x_points[i]
    
    for j in range(0, num_of_points):
        matrix4[2*j][1] = y_points[j]
        matrix4[2*j + 1][1] = y_points[j]

    for k in range(0, num_of_points):
        matrix4[2*k + 1][2] = slopes[k]
        
    size = len(matrix4)
    
    for i in range(2, size):
        for j in range(2, i+2):

            if j >= len(matrix4[i]) or matrix4[i][j] != 0:
                continue
            
            left = matrix4[i][j-1]
            
            diagonal_left = matrix4[i-1][j-1]
            
            numerator = left - diagonal_left
            
            denominator = matrix4[i][0] - matrix4[i-j+1][0]
            
            operation = numerator / denominator
            
            matrix4[i][j] = operation
            
    return matrix4

x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
slopes = [-1.195, -1.188, -1.182]

ans_4 = hermite_interpolation(x_points, y_points, slopes)

print(ans_4, "\n")

# ========== QUESTION 5A ==========

def matrix_A(x_points, y_points):
    
    numPoints = len(x_points)
    matrix_5a = numpy.zeros((numPoints, numPoints))
    
    matrix_5a[0][0] = matrix_5a[numPoints-1][numPoints-1] = 1
    
    for i in range (0, numPoints):
        for j in range (0, numPoints):
            
            if i - j == -1 and i != 0:
                    matrix_5a[i][j] = x_points[j] - x_points[i]
                
            if i - j == 1 and i != numPoints - 1:
                matrix_5a[i][j] = x_points[i] - x_points[j]
            
    for k in range (1, numPoints - 1):
        
        matrix_5a[k][k] = 2*(matrix_5a[k][k-1] + matrix_5a[k][k+1])
    
    return matrix_5a

x_points = [2, 5, 8, 10]
y_points = [3, 5, 7, 9]

ans_5a = matrix_A(x_points, y_points)

print(ans_5a, "\n")

# ========== QUESTION 5B ==========

def vector_b(x, y):
    
    b = numpy.zeros(len(x))
    
    for i in range(1, len(x) - 1):
        
        b[i] = (3/(x[i+1] - x[i]))*(y[i+1] - y[i]) - (3/(x[i] - x[i-1]))*(y[i] - y[i-1])
    
    return b

ans_5b = vector_b(x_points, y_points)

print(ans_5b, "\n")

# ========== QUESTION 5C ==========

def vector_x(A, b):
    
    return numpy.dot(numpy.linalg.inv(A), b)
    
ans_5c = vector_x(ans_5a, ans_5b)

print(ans_5c, "\n")
