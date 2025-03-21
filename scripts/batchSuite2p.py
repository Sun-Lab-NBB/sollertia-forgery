import json
import os

from pathlib import Path

import suite2p


ANIMAL_DIR = '/workdir/<userID>/<animal>/'
for sess_dir in os.listdir(ANIMAL_DIR):
    # if sess_dir in ('2021_12_28',):
    #     continue
    try:
        print(sess_dir)
        SESS_DIR = os.path.join(ANIMAL_DIR, sess_dir)
        OUT_DIR: str = os.path.join('/workdir/<userID>/data/processed/<animal>', sess_dir)
        with open(os.path.join(SESS_DIR, 'ops.json'), 'r', encoding='utf-8') as f:
            ops = json.load(f)

        ops.update({
            'multiplane_parallel': False,
            'delete_bin': 0,
            'combined': 1,
            'keep_movie_raw': 1,
            'force_sktiff': True,
        })

        db = {
            'data_path': [SESS_DIR],
            'save_path0': OUT_DIR,
            'tiff_list': sorted([str(p) for p in Path(SESS_DIR).rglob('*.tiff')]),
        }
        ops.update(db)

        server = dict(
            host='cbsuwsun.biohpc.cornell.edu',
            username='<userID>',
            password='<password>',
            server_root='/',
            local_root='/',
            n_cores=8,
        )

        output_ops = suite2p.run_s2p(ops=ops, db=db, server=server)

        log_file = list(Path(SESS_DIR).glob("*Log*.json"))
        with open(log_file[0], 'r', encoding='utf-8') as f:
            log = json.load(f)
            with open(os.path.join(OUT_DIR, log_file[0].name), 'w', encoding='utf-8') as f1:
                json.dump(log, f1, indent=4)
    except Exception as e:
        print(e)
