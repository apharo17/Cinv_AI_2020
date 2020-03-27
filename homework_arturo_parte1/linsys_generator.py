import numpy as np

#---Input parameters---
n=1200

sign = 2*(np.random.randint(2,size=(n,n))-.5)
A = 100*sign*np.random.rand(n,n)
sign_x = 2*(np.random.randint(2,size=n)-.5)
x = 100*sign_x*np.random.rand(n)
b = A.dot(x)

filename = 'linsys_' + str(n) + '.txt'
f=open(filename,'w')
for i in range(n):
    for j in range(n+1):
        if j == n:
            strout = '{}\n'.format(b[i])
        else:
            strout = '{} '.format(A[i][j])
        f.write(strout)
f.close()
