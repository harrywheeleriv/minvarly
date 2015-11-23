
#this module contains two functions. 
#       - a function to calculate the magnetic variance matrix, along with its eigen values and eigen vectors
#       - a second function that applies minimum variance rotation to the vector that is inputted (B_vector). 
#
# 
#   inputs
#       B_vector   - should be a (3,n) vector, that will be rotated
# 
#   outputs
#       b               - the final field, rotated into the minimum variance direction
#       angle           - angle calculated from b_0 and the min direction
#       eigen_vectors   - a matrix containing the eigenvectors of the magnetic
#                           variance matrix, in column vector format. 
# 
#                         eigenVectors(:,0) = maximum direction
#                         eigenVectors(:,1) = intermediate direction
#                         eigenVectors(:,2) = minimum direction
# 
#       eigen_values - eigen values stored in column vector [max inter min], nomalized to the minimum value
#       RMS          - Root mean square value of the rotated field in the
#                       minimum direction
# 
#   Harry Wheeler
import numpy as np


def magneticVariance(B_vector):  
    '''     
        the matrix M (magnetic variance matrix) is constructed and returned, along with its eigenvalues and eigenvectors. The eigenvalue are normalized by the minimum eigenvalue.
    '''
    M = np.zeros([3,3])
    
    for i in range(3):
        for j in range(3):
            M[i,j] = np.mean(np.multiply(B_vector[i,:],B_vector[j,:])) - np.mean(B_vector[i,:])*np.mean(B_vector[j,:])
        # M[i,0] = np.mean(np.multiply(B_vector[i,:],B_vector[0,:])) - np.mean(B_vector[i,:])*np.mean(B_vector[0,:])
        # M[i,1] = np.mean(np.multiply(B_vector[i,:],B_vector[1,:])) - np.mean(B_vector[i,:])*np.mean(B_vector[1,:])
        # M[i,2] = np.mean(np.multiply(B_vector[i,:],B_vector[2,:])) - np.mean(B_vector[i,:])*np.mean(B_vector[2,:])

    eigen_values, eigen_vectors = np.linalg.eig(M)     
    #sorting the vectors so max = 0, inter = 1, min = 2
    group = []
    for index, row in enumerate(eigen_values):
        group.append([row,eigen_vectors[:,index]])

    group.sort(key = lambda row: row[:][0])

    del eigen_vectors
    for index, row in enumerate(group):
        eigen_values[index] = (group[2-index][0])
        try:
            eigen_vectors = np.concatenate((eigen_vectors, group[2-index][1].reshape(3,1)), axis = 1)
        except:
            eigen_vectors = group[2-index][1].reshape(3,1)


    return M, eigen_values/eigen_values[2], eigen_vectors, group

def minvarRotate(B_vector):   
    ''' 
        This function calls the magneticVariance function which generates the variance matrix then it rotates in inputted data
        into minimum variance coordinates. It also returns the angle between the minimum variance direction and B0.
    '''

    M, eigenValues, eigenVectors, grouping = magneticVariance(B_vector)
    
    b0Direction = np.mean(B_vector,axis=1)/np.linalg.norm(np.mean(B_vector,axis=1))

    print(eigenVectors[:,2])
    angle = 180/np.pi*np.arccos(np.dot(eigenVectors[:,2],b0Direction))
    b = np.zeros([3,np.shape(B_vector[1,:])[1]])
    for i in range(np.shape(B_vector[1,:])[1]):
        b[0,i]=eigenVectors[0,0]*B_vector[0,i] + eigenVectors[1,0]*B_vector[1,i] + eigenVectors[2,0]*B_vector[2,i]
        b[1,i]=eigenVectors[0,1]*B_vector[0,i] + eigenVectors[1,1]*B_vector[1,i] + eigenVectors[2,1]*B_vector[2,i]
        b[2,i]=eigenVectors[0,2]*B_vector[0,i] + eigenVectors[1,2]*B_vector[1,i] + eigenVectors[2,2]*B_vector[2,i]
    
    RMS = np.sqrt( sum( np.power(b[2,:],2) ) / len(b[2,:]) );

    return b, angle, RMS, eigenValues, eigenVectors
