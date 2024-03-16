import numpy as np
import os
import kymatio.numpy as kn

class Preprocessing:
    def __init__(self):
        self.powerspectra={}
    
    def dictionary_collect(self, object_path, bin_factor=100, BH=True):
        """
        This function loads on the self.powerspectra dictionary the observations
        in a BH or NS directory. 
    
        Parameters:
        - object_path: directory where black holes or neutron star observations are located.
        - bin_factor: specifies which binned files should be read.
    
        Returns:
        None.
        """
        for source in os.listdir(object_path):
            for observation in os.listdir(os.path.join(object_path,source)):
                observation_path = os.path.join(object_path, source, observation, 'pca')

                # Check if observation_path is a directory
                if os.path.isdir(observation_path):

                    # List all rebinned .asc files in the observation_path
                    list_rebinned_PS = [file for file in os.listdir(observation_path) if file.endswith('.asc_' + str(bin_factor))]
                    # Iterate over each .asc file
                    for spectrum_file in list_rebinned_PS:
                        self.powerspectra.update({spectrum_file:{
                                                                'observation':[],
                                                                'Type':[]
                                                                    }})
                        binned_powerspectra_file=os.path.join(observation_path, spectrum_file)
                        tmp_spectrum_file = np.loadtxt(binned_powerspectra_file, skiprows=12)
                        # Add a new target column to write down whether we have a BH or not
                        self.powerspectra[spectrum_file]['Type'] = 1 if BH else 0
                        self.powerspectra[spectrum_file]['observation'].append(tmp_spectrum_file)
                    else:
                        continue
                else:
                    continue
        # Convert the list of result arrays into a single NumPy array
        #return np.vstack(result_arrays_list)
        return 
    
    def build_train_test_from_dict(self,):
        x=[]
        y=[]
        for observation in self.powerspectra.keys():
            x.append(self.powerspectra[observation]['observation'])
            y.append(self.powerspectra[observation]['Type'])
        return np.vstack(x),np.array(y)
        
    
    def array_collect(self, object_path, bin_factor=100, BH=True):
        result_arrays_list = []
        for source in os.listdir(object_path):
            for observation in os.listdir(os.path.join(object_path,source)):
                observation_path = os.path.join(object_path, source, observation, 'pca')

                # Check if observation_path is a directory
                if os.path.isdir(observation_path):

                    # List all rebinned .asc files in the observation_path
                    list_rebinned_PS = [file for file in os.listdir(observation_path) if file.endswith('.asc_' + str(bin_factor))]
                    # Iterate over each .asc file
                    for spectrum_file in list_rebinned_PS:
                        binned_powerspectra_file=os.path.join(observation_path, spectrum_file)
                        tmp_spectrum_file = np.loadtxt(binned_powerspectra_file, skiprows=12)
                        # Add a new target column to write down whether we have a BH or not
                        new_column = np.ones(tmp_spectrum_file.shape[0]) if BH else np.zeros(tmp_spectrum_file.shape[0])
                        new_column_reshaped = new_column.reshape(-1, 1)
                        result_array = np.hstack((tmp_spectrum_file, new_column_reshaped))
                        result_arrays_list.append(result_array)
                    else:
                        continue
                else:
                    continue
        # Convert the list of result arrays into a single NumPy array
        return np.vstack(result_arrays_list)

    
def compute_transform(X, nodes):
    n = int(np.log(nodes)/np.log(2))
    
    T=2**n
    J = 3  # Number of scales
    Q = 8  # Number of wavelets per octave


    scattering = kn.Scattering1D(J,T,Q)
    
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)
    
    Sx = scattering(X)
    
    return Sx[order0][0],Sx[order1],Sx[order2]
        
