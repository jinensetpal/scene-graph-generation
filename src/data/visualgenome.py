#!/usr/bin/env python3

from dagshub.data_engine import datasources, datasets
from torch_geometric.data import HeteroData
from ..utils import ohe
from glob import glob
from .. import const
import pandas as pd
import torchvision
import dagshub
import random
import torch
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.images = glob(str(const.DATA_DIR / 'images' / '*'))
        self.relationships = json.load(open(str(const.DATA_DIR / 'relationships.json')))
        self.get_graph = GraphGeneration(json.load(open(str(const.DATA_DIR / 'object_synsets.json'))),
                                         json.load(open(str(const.DATA_DIR / 'relationship_synsets.json')))).get_graph

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return image_norm(self.images[idx]), self.get_graph(str(self.relationships[idx]))


class GraphGeneration:
    def __init__(self, obj_map, rel_map):
        self.obj_map = obj_map
        self.rel_map = rel_map

        self.obj_all = list(set(obj_map.values()))
        self.rel_all = list(set(rel_map.values()))

        self.n_obj = len(self.obj_all)
        self.n_rel = len(self.rel_all)

    def get_graph(self, relationships):
        obj = []
        rel = []

        for pair in eval(relationships)['relationships']:
            try:
                x = ((self.obj_map[pair['subject']['name']], pair['subject']['x'], pair['subject']['y'], pair['subject']['h'], pair['subject']['w']), self.rel_map[pair['predicate']], (self.obj_map[pair['object']['name']], pair['object']['x'], pair['object']['y'], pair['object']['h'], pair['object']['w']))
                obj.extend((x[0], x[2]))
                rel.append(x[1])
            except KeyError:
                print('Failed to identify:', pair['subject']['name'], pair['predicate'], pair['object']['name'])
        df = pd.DataFrame(set(obj))

        graph = HeteroData()
        graph['object'].boxes = torch.tensor(df[range(1, 5)].to_numpy())
        graph['object'].id = torch.vstack(list(map(lambda x: ohe(self.n_obj, self.obj_all.index(x)), df[0])))

        graph['relation'].id = torch.vstack(list(map(lambda x: ohe(self.n_rel, self.rel_all.index(x)), rel)))

        graph['object', 'to', 'relation'].edge_index = torch.arange(0, len(rel)).unsqueeze(1).repeat(1, 2).t()
        graph['relation', 'to', 'object'].edge_index = torch.arange(0, len(rel)).unsqueeze(1).repeat(1, 2).t()
        graph['relation', 'to', 'object'].edge_index[1, :] += 1

        return graph


def image_norm(filepath):
    return torchvision.io.read_image(filepath) / 255


def enrich(row):
    row['type'] = 'image' if row['path'].startswith('images/') else 'metadata'
    if row['type'] == 'image': row['id'] = int(row['path'].split('/')[-1].split('.')[0])
    return row


def preprocess(manual=True):
    if not len(datasources.get_datasources(const.REPO_NAME)):
        # you may have to wait until datasource scanning is completed, otherwise nothing will be preprocessed.
        ds = datasources.create_from_bucket(const.REPO_NAME,
                                            const.DATASOURCE_NAME,
                                            const.BUCKET_NAME)

    else: ds = datasources.get_datasource(const.REPO_NAME, name=const.DATASOURCE_NAME)

    if not manual:
        metadata = datasources.get_datasource('ML-Purdue/sgg-template', 'visualgenome').all().download_binary_columns('relationships', load_into_memory=True).dataframe  # do not change this line, hardcoded on purpose

        dagshub.common.config.dataengine_metadata_upload_batch_size = 500  # optional if you want a smaller upload size (recommended if you have a lot of metadata per file)
        ds.upload_metadata_from_dataframe(metadata, path_column='path')
    else:
        print('[INFO] Enriching dataset.')
        df = ds.all().dataframe
        df = df[df['path'].apply(lambda x: x.startswith('images/'))].apply(enrich, axis=1)
        ds.upload_metadata_from_dataframe(df)

        print('[INFO] Obtaining metadata.')
        pred = (ds['type'] == 'metadata').all().as_ml_dataset('torch', tensorizers=[pd.read_json])[13][0]
        print('[INFO] Merging frames.')
        df = pd.DataFrame.merge(df, pred,
                                left_on='id', right_on='image_id', how='outer')
        df['relationships'] = df['relationships'].fillna('').apply(list).apply(str)
        df = df.drop('image_id', axis=1)

        print('[INFO] Adding splits.')
        df['split'] = random.choices(list(const.SPLITS.keys()),
                                     weights=list(const.SPLITS.values()),
                                     k=len(df))

        print('[INFO] Uploading metadata.')
        ds.upload_metadata_from_dataframe(df)

    print('[INFO] Saved as a dataset.')
    (df['type'] == 'image').save_dataset(const.DATASET_NAME)


def get_generators(force_preprocessing=False):
    if force_preprocessing or not len(datasets.get_datasets(const.REPO_NAME)): preprocess()
    ds = datasets.get_dataset(const.REPO_NAME, const.DATASET_NAME)
    metaset = (datasources.get_datasource(const.REPO_NAME, const.DATSOURCE_NAME)['type'] == 'metadata').all().as_ml_dataset('torch', tensorizers=[lambda x: json.load(open(x)),])

    kwargs = {'flavor': 'torch',
              'shuffle': True,
              'strategy': 'background',
              'batch_size': const.training.BATCH_SIZE,
              'metadata_columns': ['relationships',],
              'tensorizers': [image_norm, GraphGeneration(metaset[2], metaset[14]).get_graph]}
    return [(ds['split'] == split).all().as_ml_dataloader(**kwargs) for split in ['train', 'val', 'test']], metaset[2]


if __name__ == '__main__':
    train, valid, test = get_generators(force_preprocessing=True)
