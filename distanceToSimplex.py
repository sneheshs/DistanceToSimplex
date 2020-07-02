import numpy as np
from sympy import Matrix
from frref.frref import frref

def distanceToSimplex(point, S):
    '''
    An implementation of An Algorithm to Compute the Distance from a Point to a Simplex
    by Oleg Golubitsky, Vadim Mazalov and Stephen M. Watt

    Matlab Implementation: Douglas Summers Stay (https://github.com/summerstay)
    Python Implementation: Snehesh Shrestha (https://github.com/sneheshs)
    '''

    n = np.shape(S)[0]

    if n == 1:
        distance = np.linalg.norm(point-S[0])
        projection = np.matrix(S[0]).T

    else:
        translatedS = S-S[0]        

        b = np.matrix((point-S[0])).T
        A = np.matrix((translatedS[1:])).T

        top = np.dot(A.T, A)
        #top = top+.00000001*np.random.rand(np.shape(top)[0], np.shape(top)[1])

        bottom = np.dot(A.T, b)
        #bottom = bottom+.00000001*np.random.rand(np.shape(bottom)[0])

        # divide
        alpha = np.linalg.solve(top, bottom)

        # # SLOW RREF
        # __A = Matrix(A)
        # _, pivcol = __A.rref()  # frref(A)
        # FAST RREF
        _, pivcol = frref(A)
        A = A[:, pivcol]

        # P = A*inv(A'*A)*A'
        P = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
        pprime = np.dot(P, b)

        posflag = 1
        # for ii in range(0, np.shape(alpha)[0]):
        #     if alpha[ii] < 0:
        if any(alpha < 0):
            posflag = 0

        if sum(alpha) <= 1 and posflag == 1:
            # projection inside simplex
            distance = np.linalg.norm(pprime-b)
            projection = np.matrix(pprime+S[0]).T

        elif posflag == 0:
            Sprime = []

            #Sprime[0] = S[0]
            Sprime.append(S[0])

            for ii in range(0, np.shape(alpha)[0]):
                if alpha[ii] > 0:
                    Sprime.append(S[ii+1])
                    
            distance, projection = distanceToSimplex(point, np.array(Sprime))
        else:
            distance, projection = distanceToSimplex(point, np.array(S[1:]))

    return distance, projection
