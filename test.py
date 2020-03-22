import numpy as np
import matplotlib.pyplot as plt

def fitness_analyse(self):
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.plot(A,range(5))
    plt.show()


DNA_size=5
A=[1,2,3,4,5]
A=np.array(A)
B=[4,3,2,5,1]
B=np.array(B)
cross_points = np.random.randint(0, 2, DNA_size).astype(np.bool)
print(cross_points)
idx=[i for i,v in enumerate(cross_points)if v==True]
print('idx',idx)
print(~cross_points)
print(A)
print(B)
keep=A[~cross_points]
print('keep',keep)
swap=B[np.isin(B,keep,invert=True)]
print('swap',swap)
A[idx]=swap
print(A)
fitness_analyse(A)
