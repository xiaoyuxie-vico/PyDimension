'''
Load the raw data from flow3D (vorticity equation)
and convert the data into 3D shape
You can decide whether to normalize the data before store it into npy files
'''

import sys
sys.path.append('/home/xie/projects/DimensionNet')

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')
plt.rcParams["font.family"] = "Arial"
np.set_printoptions(suppress=True)

# ROOT = '/mnt/data/dataset/cylinder'
ROOT = '/mnt/data/dataset/three_cylinder'

def parse_data(dataset_dir, file_name, interval):
    '''
    convert txt file to csv and save
    '''
    data_all = []
    file_path = f'{dataset_dir}/{file_name}'
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        Nt = int(len(lines) / interval)

        print('Nt, interval', Nt, interval)
        # loop for each time step
        for i in tqdm(range(Nt)):
            # loop for each node
            for j in range(interval):
                index = interval * i + j
                # remove heading and the last blank line
                if j not in range(5):
                    str_split = lines[index].split(' ')
                    str_clean = [float(each) for each in str_split if each != '']
                    data_all.append([i] + str_clean)

    df = pd.DataFrame(data_all, columns=['Nt', 'x', 'y', 'z', 'u'])
    df.to_csv(file_path.replace('.txt', '.csv'))
    print(df.head())
    print(df.shape)


class NSDataset(object):
    
    def __init__(self, case_id, Nx, Nz, dt, file_list, is_normalize=False):
        super(NSDataset, self).__init__()
        self.case_id = case_id
        self.dataset_dir = f'{ROOT}/{case_id}'
        self.interval = Nx * Nz
        self.file_list = file_list
        self.is_normalize = is_normalize
        self.Nx, self.Nz = Nx, Nz
        self.Nt = self.cal_Nt()
        self.dt = dt

        # # read data and combine them
        self.df = self.combine_data()
        # self.df.to_csv(f'{ROOT}/{self.case_id}/combined_data_no_normalization.csv')
        
        self.u_3D, self.v_3D, self.w_3D, self.p_3D = self.rearrange()

        np.save(f'{ROOT}/{self.case_id}/u_3D.npy', self.u_3D)
        np.save(f'{ROOT}/{self.case_id}/v_3D.npy', self.v_3D)
        np.save(f'{ROOT}/{self.case_id}/w_3D.npy', self.w_3D)
        np.save(f'{ROOT}/{self.case_id}/p_3D.npy', self.p_3D)
        
    def cal_Nt(self):
        file_path = f'{self.dataset_dir}/Pressure.txt'
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
            Nt = int(len(lines) / interval)
        return Nt

    def combine_data(self):
        '''
        Combine each simulation case results in one dataFrame
        '''
        res = {}
        print('Start combine data into one dataframe')
        for file_name in file_list:
            print('Parsing', file_name)
            file_path = f'{self.dataset_dir}/{file_name}'
            df_each = pd.read_csv(file_path)
            res[file_name.split('.')[0]] = df_each

        df = res['Velocity_X']
        df['v'] = res['Velocity_Y']['u']
        df['w'] = res['Vorticity_Z']['u']
        df['t'] = df['Nt'] * self.dt
        df['p'] = res['Pressure']['u']

        return df
        
    def rearrange(self):
        '''
        Rearrange data in 3D
        '''
        u, v, w, p = self.df['u'].to_numpy(), self.df['v'].to_numpy(), self.df['w'].to_numpy(), self.df['p'].to_numpy()
        u_3D = u.reshape(self.Nt, self.Nz, self.Nx)
        v_3D = v.reshape(self.Nt, self.Nz, self.Nx)
        w_3D = w.reshape(self.Nt, self.Nz, self.Nx)
        p_3D = p.reshape(self.Nt, self.Nz, self.Nx)
        return u_3D, v_3D, w_3D, p_3D
   

if __name__ == '__main__':
    # case_id_list = [
    #     # 'Re-115', # done
    #     # 'Re-122', # done
    #     # 'Re-127', # done
    #     # 'Re-136', # done
    #     # 'Re-148', # done
    #     # 'Re-165', # done
    #     # 'Re-171', # done
    #     # 'Re-185',  # done
    #     # 'Re-50',  # done
    #     # 'Re-50-v2', # done
    #     # 'Re-191',  # done
    #     # Stage 2 dataset
    #     # 'v2-Re-50',  # done
    #     # 'v2-Re-100', # done
    #     'v2-Re-70', 
    #     'v2-Re-150',
    #     'v2-Re-200',
    # ]
    case_id_list = [
        # 'v2-Re-50', # done
        # 'v2-Re-70', 
        # 'v2-Re-80', 
        # 'v2-Re-90', 
        # 'v2-Re-100', 
        'v1-Re-120',
        'v1-Re-150', 
        'v1-Re-170', 
        'v1-Re-190',
        'v1-Re-200',  
    ]
    for case_id in case_id_list:
        print('*'*40)
        print(f'case_id: {case_id}')
        Nx, Nz = 500, 222
        # dt = 0.000004
        dt = 0.00004
        interval = Nx * Nz + 5  # 5 extra lines for each time step
        
        file_list = ['Pressure.txt', 'Velocity_X.txt', 'Velocity_Y.txt', 'Vorticity_Z.txt']
        dataset_dir = f'{ROOT}/{case_id}'
        for file_name in file_list:
            print('file_name', file_name)
            parse_data(dataset_dir, file_name, interval)
        
        # ###############################Create library###############################
        file_list = ['Pressure.csv', 'Velocity_X.csv', 'Velocity_Y.csv', 'Vorticity_Z.csv']
        dataset = NSDataset(case_id, Nx, Nz, dt, file_list, is_normalize=False)
