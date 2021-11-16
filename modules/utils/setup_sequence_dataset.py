import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from .setup_dataset import _process_aug, _process


def _process_gen_items(tensor_item, mode, out_mode, batch_size=1):
    if mode == 'train':
        if out_mode == 'meta':
            ds = tf.data.Dataset.from_tensor_slices(({
                'meta_0':
                tensor_item[0]['meta_0'].astype('float32')
            }, tensor_item[1].astype('float32'))).batch(batch_size)
        if out_mode == 'image':
            ds = tf.data.Dataset.from_tensor_slices(({
                'img_0':
                tensor_item[0]['img_0'].astype('float32')
            }, tensor_item[1].astype('float32')))
            ds = ds.map(_process_aug).batch(batch_size)
        if out_mode == 'combined':
            ds = tf.data.Dataset.from_tensor_slices(({
                'meta_0':
                tensor_item[0]['meta_0'].astype('float32'),
                'img_0':
                tensor_item[0]['img_0'].astype('float32')
            }, tensor_item[1].astype('float32')))
            ds = ds.map(_process_aug).batch(batch_size)
    else:
        if out_mode == 'meta':
            ds = tf.data.Dataset.from_tensor_slices(({
                'meta_0':
                tensor_item[0]['meta_0'].astype('float32')
            }, tensor_item[1].astype('float32'))).batch(batch_size)
        if out_mode == 'image':
            ds = tf.data.Dataset.from_tensor_slices(({
                'img_0':
                tensor_item[0]['img_0'].astype('float32')
            }, tensor_item[1].astype('float32')))
            ds = ds.map(_process).batch(batch_size)
        if out_mode == 'combined':
            ds = tf.data.Dataset.from_tensor_slices(({
                'meta_0':
                tensor_item[0]['meta_0'].astype('float32'),
                'img_0':
                tensor_item[0]['img_0'].astype('float32')
            }, tensor_item[1].astype('float32')))
            ds = ds.map(_process).batch(batch_size)
    return ds


def index_map(csv_file):
    df = pd.read_csv(csv_file)

    idx_map = {}
    for row_idx in df.index:
        part_id = df.loc[row_idx, 'participant_id']
        scan_date = df.loc[row_idx, 'scan_date']
        idx_map[row_idx] = [part_id, scan_date]

    return idx_map


def write_embeddings(generator,
                     mode,
                     out_mode,
                     model,
                     index_map,
                     save_dir,
                     nb_eps=1):
    emb_ds = {}

    for ep in range(nb_eps):
        print('Epoch:', ep)
        data = {}

        for item, r_idx in tqdm(generator):
            inp, lbl = item
            ds = _process_gen_items([inp, lbl], mode, out_mode)
            emb = np.squeeze(model.predict(ds))
            for idx in range(len(r_idx)):
                r_tmp = r_idx[idx]
                part_id, sdate = index_map[r_tmp]
                try:
                    data[part_id][sdate] = [emb[idx], lbl[idx]]
                except KeyError:
                    data[part_id] = {sdate: [emb[idx], lbl[idx]]}
        emb_ds[ep] = data

    with open(save_dir, 'wb') as handle:
        pickle.dump(emb_ds, handle)
