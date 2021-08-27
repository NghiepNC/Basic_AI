import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

def cost(x):
	m= A.shape[0]
	return 0.5/m *np.linalg.norm (A.dot(x)-b,2)**2
def grad(x):
	m= A.shape[0]
	return 1/m * A.T.dot(A.dot(x)-b)

def gardient_desecent(x_init, learning_rate, interation):
	x_list = [x_init]
	for i in range (interation):
		x_new  = x_list[-1]- learning_rate*grad(x_list[-1])
		m = len(x_new)
		if np.linalg.norm(grad(x_new))/m <0.3:  #stop GD
			break
		x_list.append(x_new)
	return x_list
def check_grad(x):
	eps = 1e-4
	g = np.zeros_like(x)
	for i in range (len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		# gradient coong thuc kiem tra đạo hàm
		g[i] = (cost(x1)-cost(x2))/(2*eps)

	# Ham gradient tự tính
	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-5:
		print("WARNING: CHECK GRADIENT FUNCTION!")


# Data
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
fig1 = plt.figure("GD for gi gi do^^")
ax = plt.axes(xlim = (-10,50), ylim = (-10,50))
plt.plot(A, b, "ro")

#tạo ma trận dọc giá trị là 1
ones =  np.ones((A.shape[0],1), dtype= np.uint8)
#create A square
x_square = np.array(A[:]**2)
# 2 ma trận ghép song song
A = np.concatenate((x_square,A, ones), axis =1) 


#line linear regression 
lr = linear_model.LinearRegression()
lr.fit(A,b)
x0 = np.linspace(1,46,10000)
y0_sklearn= lr.coef_[0][0]*x0*x0+ lr.coef_[0][1]*x0 + lr.intercept_[0]
plt.plot(x0,y0_sklearn, color = "green")



# Random initial line y = -2x^2 +3x-2
# x_init =  np.array([[-2.],[3.], [-2.]])
x_init =  np.array([[-2.],[5.], [-2.]])
y0_init = x_init[0][0]*x0*x0+ x_init[1][0]*x0+ x_init[2][0]
plt.plot(x0, y0_init, color = "black")

# kiem tra dao ham
check_grad (x_init)

# vẽ Gradient Descent
learning_rate = 0.000001
interation =90
x_list = gardient_desecent(x_init, learning_rate, interation)
for i in range (len(x_list)):
	y0_x_list = x_list[i][0]*x0*x0+ x_list[i][1]*x0  + x_list[i][2]
	plt.plot(x0, y0_x_list, color = "black",alpha = 0.3)


# Draw animation
line , = ax.plot([],[], color = "blue")
def update(i):
	y0_gd = x_list[i][0][0]*x0*x0+ x_list[i][1][0]*x0+ x_list[i][2][0]
	line.set_data(x0, y0_gd)
	return line,

iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

# legend for plot
plt.legend(('Value in each GD iteration', 'Solution by formular', 'Inital value for GD'), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()

# title
plt.title("Gradient Descent Animation")
plt.show()

# #bieu do cost tren interation

# cost_list = []
# iter_list = []
# for i in range (len(x_list)):
# 	iter_list.append(i)
# 	cost_list.append(cost(x_list[i]))

# plt.plot(iter_list,cost_list)
# plt.xlabel("iteration")
# plt.ylabel("cost value")
# plt.show()