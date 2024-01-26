import numpy as np
from sklearn.cluster import AgglomerativeClustering


def get_x_max (nx_group_1, nx_group_2):
    '''x_avg calculation (X_AVG)
    Find x_avg as parameter for region identification
    Param in:   nx_group_1
                nx_group_2
    Param out:  x_avg

    '''
    # Get pointcloud closer to the origin
    if np.max(nx_group_1) < np.max(nx_group_2):
        x_max = np.max(nx_group_1)
    elif np.max(nx_group_1) > np.max(nx_group_2):
        x_max = np.max(nx_group_2)
    elif np.max(nx_group_1) == np.max(nx_group_2):
        assert False, "[ERR][X_AVG] T.B.D"

    return x_max
    
def ah_clustering_dense(pcl):
    '''Agglomerative hierarchical clustering (AHC)
    Perform clustering based on distance of points
    Param in:   pcl     input pointcloud with dimension (x, y, z, r)
    Param out:  labels  label fro each points

    '''
    #--------------------------------------------------------------------------
    # REPARE
    #--------------------------------------------------------------------------
    # Get Pointcloud Data
    pcl = np.fromfile(pcl, dtype=np.float32).reshape(-1, 5)

    # Get label from input Pointcloud   
    label = pcl[:, 4]
  
    # Get points labeled as "noise"                                      
    # nx, ny, nz = pcl[(label == 1), :3].T 
    nx, ny, nz, nr = pcl[(label == 1), :4].T 
    
    # noise_points = np.array([nx]).T
    # noise_points = np.array([nx, ny, nz]).T
    noise_points = np.array([nx, ny, nz, nr]).T

    # Iterative parameter
    num_points = len(noise_points) 

    if num_points == 0:
        return None, None, None

    # Check error
    assert  ((nx.shape[0] == len(noise_points))  & \
             (ny.shape[0] == len(noise_points))  & \
             (nz.shape[0] == len(noise_points))) , \
            "[ERR][ah_clustering_dense][PREPARE] Incompatible shape"                                                                

    #--------------------------------------------------------------------------
    # AHC
    #--------------------------------------------------------------------------
    ahc = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = ahc.fit(noise_points).labels_

    # Check error
    assert  ((nx.shape[0] == labels.shape[0])  & \
             (ny.shape[0] == labels.shape[0])  & \
             (nz.shape[0] == labels.shape[0])) , \
            "[ERR][ah_clustering_dense][AHC] Incompatible shape"
    
    # Get x coordinate of noise Point groups 
    nx_group_1 = [nx[x] for x in range(num_points) if labels[x] == 1]
    nx_group_2 = [nx[x] for x in range(num_points) if labels[x] == 0]
    
    return labels, nx_group_1, nx_group_2

