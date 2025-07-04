"""
Modified from: https://github.com/CalciferZh/SMPL/blob/master/preprocess.py
"""


import inspect
# Patch for Python 3.11+ to bring back getargspec using getfullargspec
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


import os
import numpy as np
import pickle
import sys

if __name__ == '__main__':
    src_paths = ['./data_SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl',
                './data_SMPL/basicmodel_m_lbs_10_207_0_v1.0.0.pkl']

    output_dir = './data_SMPL'
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."

    for src_path in src_paths:
        bname = os.path.basename(src_path)
        if '_f_' in bname:
            output_path = os.path.join(output_dir, 'model_f.pkl')
        elif '_m_' in bname:
            output_path = os.path.join(output_dir, 'model_m.pkl')
        else:
            assert NotImplementedError(f"Unknown model: {bname}")

        with open(src_path, 'rb') as f:
            src_data = pickle.load(f, encoding="latin1")
        print(f'loaded {src_path}:')
        for k, v in src_data.items():
            print(f'  - {k}: {v.shape if isinstance(v, np.ndarray) else type(v)}')
        model = {
            'J_regressor': src_data['J_regressor'],
            'weights': np.array(src_data['weights']),
            'posedirs': np.array(src_data['posedirs']),
            'v_template': np.array(src_data['v_template']),
            'shapedirs': np.array(src_data['shapedirs']),
            'f': np.array(src_data['f']),
            'kintree_table': src_data['kintree_table']
        }
        if 'cocoplus_regressor' in src_data.keys():
            model['joint_regressor'] = src_data['cocoplus_regressor']
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Processed {bname} and saved to {output_path}\n")
    
    print("Done")