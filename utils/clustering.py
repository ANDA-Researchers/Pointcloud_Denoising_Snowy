import numpy as np
from sklearn.cluster import AgglomerativeClustering

def ah_clustering_dense(pointcloud):
    '''
    Agglomerative hierarchical clustering (AHC)

    '''
    #-------------
    # REPARE
    #-------------
    # Get Pointcloud Data
    pointcloud = np.fromfile(pointcloud, dtype=np.float32).reshape(-1, 5)

    # Get label from input Pointcloud   
    label = pointcloud[:, 4]
  
    # Get points labeled as "noise"                                      
    nx, ny, nz = pointcloud[(label == 1), :3].T 
    
    # noise_points = np.array([nx]).T
    noise_points = np.array([nx, ny, nz]).T

    # Iterative parameter
    num_points = len(noise_points) 

    if num_points == 0:
        return None, None, None

    # Check error
    assert  ((nx.shape[0] == len(noise_points))  & \
             (ny.shape[0] == len(noise_points))  & \
             (nz.shape[0] == len(noise_points))) , \
            "[ERR][ah_clustering_dense][PREPARE] Incompatible shape"                                                                

    #-------------
    # AHC
    #-------------
    ahc = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = ahc.fit(noise_points).labels_

    # Check error
    assert  ((nx.shape[0] == labels.shape[0])  & \
             (ny.shape[0] == labels.shape[0])  & \
             (nz.shape[0] == labels.shape[0])) , \
            "[ERR][ah_clustering_dense][AHC] Incompatible shape"
    
    # Get Point groups
    nx_group_1 = [nx[x] for x in range(num_points) if labels[x] == 1]
    nx_group_2 = [nx[x] for x in range(num_points) if labels[x] == 0]
    
    return labels, nx_group_1, nx_group_2