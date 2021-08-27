import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# # random data dạng đường thẳng
# A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
# b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#DATA dạng parapol
b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]
A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
plt.plot(A,b,'ro')

#transfrom thành ma trận dọc
A= np.array([A]).T 
b = np.array ([b]).T 


#create A square
x_square = np.array(A[:]**2)


#tạo ma trận dọc giá trị là 1
ones =  np.ones((A.shape[0],1), dtype= np.uint8)

# 2 ma trận ghép song song
A = np.concatenate((x_square,A, ones), axis =1) 

#tính theo công thức tìm a,b
x= np.linalg.inv(A.T .dot(A)).dot(A.T.dot(b))

#tạo dữ liệu đầu cuối để vẽ đường thẳng
x0= np.linspace(1,25,1000)
y0= x[0][0]*x0*x0 + x[1][0]*x0 + x[2][0]

plt.plot(x0,y0)

# Vualize data
plt.show()