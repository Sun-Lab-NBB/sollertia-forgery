import os

from natsort import natsorted
from pathlib import Path

import numpy as np
import suite2p


sessions_list = [
    '2021_12_27/1', '2021_12_28/2', '2021_12_29/1', '2021_12_30/1',
    '2021_12_31/1', '2022_01_03/1', '2022_01_04/1', '2022_01_05/1',
    '2022_01_06/1'
]
ANIMAL_PROC_DIR = '/workdir/<userID>/data/processed/<animal>/'
for sess_dir in sessions_list:
    try:
        print(sess_dir)
        SESS_DIR = os.path.join(ANIMAL_PROC_DIR, sess_dir)
        ops = np.load(os.path.join(SESS_DIR, 'suite2p/plane0/ops.npy'), allow_pickle=True)
        ops_item = ops.item()
        save_folder = os.path.join(ops_item["save_path0"], ops_item["save_folder"])
        plane_folders = natsorted([
            f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
        ])
        ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
        #### COMBINE PLANES or FIELDS OF VIEW ####
        if len(ops_paths) > 1 and ops_item["combined"] and ops_item.get("roidetect", True):
            print("Creating combined view")
            suite2p.io.combined(save_folder, save=True)

        # save to NWB
        if ops_item.get("save_NWB"):
            print("Saving in nwb format")
            suite2p.io.save_nwb(save_folder)
    except Exception as e:
        print(e)
        continue
