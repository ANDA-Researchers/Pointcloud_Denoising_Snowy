
from pathlib import Path
import numpy as np
import h5py

class modelnet_data_prep():
    def __init__(self, root_dir="KITTISnow", is_training=True, sample_points=20000, seed_centroid=50) -> None:
        '''
        '''
        self.sample_points = sample_points
        self.seed_centroid = seed_centroid
        root_dir = Path(root_dir)
        if is_training:
            sub_dir = "training"
        density = ["heavy", "light", "medium"]
        self.data_dir = (root_dir / sub_dir / "velodyne_reduced" / density[0])
        self.file_list = list(self.data_dir.rglob("*.bin"))

    def fps(self, point_cloud, num_points):
        """
        Perform farthest point sampling on a point cloud.

        Returns:
        - sampled_indices: Indices of the sampled points.
        """

        # Initialize the list of sampled indices
        sampled_indices = []

        # Randomly choose the first point
        first_point_index = np.random.choice(len(point_cloud))
        sampled_indices.append(first_point_index)

        # Calculate distances to the first point
        distances = np.linalg.norm(point_cloud - point_cloud[first_point_index], axis=1)

        for _ in range(1, num_points):
            # Choose the farthest point
            farthest_index = np.argmax(distances)
            sampled_indices.append(farthest_index)

            # Update distances based on the newly sampled point
            distances = np.minimum(distances, np.linalg.norm(point_cloud - point_cloud[farthest_index], axis=1))

        return sampled_indices


    def generate_patch(self):
        '''Generate patch
        '''
        # Collect all paths in a list
        
        centroid_input = np.empty((0, 3))  # Assuming centers have 3 coordinates (adjust accordingly)
        noise_input = np.empty((0, 3))  # Assuming noise has 3 coordinates
        gt_input = np.empty((0, 3))  # Assuming gt has 3 coordinates
        weight_input = np.empty((0, 1))  # Assuming weight is a scalar value

        for i, path in enumerate(self.file_list):
            print("iteration %d" % i)
            if i == 0:
                centers, noise, gt, weight = self.__gen_one_sample__(path)
                centroid_input = centers
                noise_input = noise
                gt_input = gt
                weight_input = weight
            else:
                centers, noise, gt, weight = self.__gen_one_sample__(path)
                centroid_input = np.concatenate((centroid_input, centers), axis=0)
                noise_input = np.concatenate((noise_input, noise), axis=0)
                gt_input = np.concatenate((gt_input, gt), axis=0)
                weight_input = np.concatenate((weight_input, weight), axis=0)
                if i == 10:
                    break
        return centroid_input, noise_input, gt_input, weight_input
    
    def create_h5(self, name="test3.h5"):
        '''Create h5 dataset
        '''
        centroid_input, noise_input, gt_input, weight_input = self.generate_patch()

        # Creating an HDF5 file
        with h5py.File("test2.h5", 'w') as f:
            # Create datasets in the HDF5 file
            f.create_dataset('center', data=centroid_input)
            f.create_dataset('noise', data=noise_input)
            f.create_dataset('noise_gt', data=gt_input)
            f.create_dataset('weight', data=weight_input)

    def __gen_one_sample__(self, data):
        data = np.fromfile(data, dtype=np.float32).reshape(-1, 5)

        idx = self.fps(data, self.sample_points)
        data = data[idx]

        data2 = np.expand_dims(data, axis=0)#[1,:,3]

        noise = data2[:,:,0:3]
        label = data2[:,:, 4]
        gt = data2[label==2,0:3]
        idx_gt = self.fps(gt, self.sample_points)
        gt = gt[idx_gt]
        gt = np.expand_dims(gt, axis=0)
        weight = np.ones((1, self.sample_points))

        # Generate seed points using farthest point sampling
        seed_point_indices = self.fps(gt, self.seed_centroid)
        seed_points = gt[seed_point_indices]
        centers = np.expand_dims(seed_points, axis=0)#[1,:,3]

        centers = np.expand_dims(seed_points, axis=0)[:,:,0:3]


        return centers, noise, gt, weight

    
def main()-> None:
    data_prep = modelnet_data_prep()
    data_prep.create_h5(name="test3.h5")

def load_h5_data( num_center=50 ,num_point = 20000,h5_filename='./test3.h5'):

    print("input h5 file is:", h5_filename)
    f = h5py.File(h5_filename)
    center=np.array(f['center'])[:,:num_center,:]
    input = np.array(f['noise'])[:,:num_point,:]
    gt = np.array(f['noise_gt'])[:,:num_center*2,:3]
    gt2	= np.array(f['noise_gt'])[:,:num_point,:3]
    weight = np.array(f['weight'])[:,:num_point]
    #assert len(center) == len(gt)

    input=np.concatenate([center,input],axis=1)

    print("Normalization the data")
    data_radius = np.ones(shape=(input.shape[0]))
    centroid = np.expand_dims(center[:, 0,0:3],axis=1)
    center[:, :, 0:3] =center[:, :,0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(center[:, :,0:3] **2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] =input[:, :,0:3] / np.expand_dims(furthest_distance, axis=-1)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    gt2[:, :, 0:3] = gt2[:, :, 0:3] - centroid
    gt2[:, :, 0:3] = gt2[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    print(len(input),len(input[0]),len(input[0][0]))
    print(len(gt),len(gt[0]),len(gt[0][0]))
    print("total %d samples" % (len(input)))
    return input, gt, gt2, data_radius, weight


if __name__ == '__main__':
    main()

