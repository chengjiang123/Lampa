import sys

sys.path.append("..")

import os
import shutil
import argparse
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from itertools import combinations, product

from torch_geometric.transforms import BaseTransform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import is_undirected, to_undirected, remove_self_loops
from joblib import Parallel, delayed
from utils import compute_edge_weight
from utils import download_url, extract_zip, decide_download


class TrackingTransform(BaseTransform):
    def __call__(self, data):
        data.edge_index = data.knn_edge_index_k60
        #data.point_pairs_index = data.point_pairs_index_rad

        data.x = torch.cat([data.x, data.layer.view(-1, 1).float() / 10.0], dim=-1)
        data.coords = torch.cat([data.pos, data.x[:, :4]], dim=-1)
        # data.edge_weight = compute_edge_weight(data)  # uncomment this line for GCN
        del data.knn_edge_index_k60, data.layer, data.sector
        return data


def get_new_idx_split(dataset):
    sorted_evtid = dataset.evtid.argsort()
    dataset_len = len(dataset)

    split = {"train": 0.7, "valid": 0.1, "test": 0.2}
    n_train = int(dataset_len * split["train"])
    n_train = n_train - n_train % 10
    n_valid = int(dataset_len * split["valid"])

    idx = sorted_evtid
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


class Tracking(InMemoryDataset):
    def __init__(self, root, dataset_name, debug=False, **kwargs):
        assert dataset_name in ["tracking-6k","tracking-3k","tracking-10k","tracking-15k","tracking-20k", "tracking-60k"]
        self.url_processed_60k = "https://zenodo.org/records/10694703/files/tracking-60k-processed.zip"
        self.url_processed_6k = "https://zenodo.org/records/10694703/files/tracking-6k-processed.zip"

        self.dataset_name = dataset_name
        #self.n_sectors = 1 if dataset_name == "tracking-60k" else 10
        if dataset_name == "tracking-20k":
            self.n_sectors = 2
        elif dataset_name == "tracking-15k":
            self.n_sectors = 3
        elif dataset_name == "tracking-6k":
            self.n_sectors = 10
        elif dataset_name == "tracking-10k":
            self.n_sectors = 6
        elif dataset_name == "tracking-3k":
            self.n_sectors = 20
        elif dataset_name == "tracking-60k":
            self.n_sectors = 1
        self.debug = debug

        self.feature_names = (
            "r",
            "phi",
            "z",
            "eta_rz",
            "u",
            "v",
            "charge_frac",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        )
        self.feature_scale = np.array(
            [1000.0, np.pi, 1000.0, 1, 1 / 1000.0, 1 / 1000.0] + [1.0] * (len(self.feature_names) - 6)
        )

        super(Tracking, self).__init__(str(root), transform=kwargs.get("transform", None), pre_transform=None)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.idx_split = get_new_idx_split(self)
        self.x_dim = self._data.x.shape[1] + 1
        self.coords_dim = 2 + 4

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw", self.dataset_name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", self.dataset_name)

    @property
    def raw_file_names(self):
    # List all raw files in the directory, filtering out directories and hidden files
        return [
            f for f in os.listdir(self.raw_dir)
            if os.path.isfile(os.path.join(self.raw_dir, f)) and not f.startswith(".")
        ]
    #def raw_file_names(self):
    #    return ["data21149_s0.pt"]

    @property
    def processed_file_names(self):
        size = self.dataset_name.split("-")[-1]
        return [f"data-{size}.pt"]

    def download(self):
        #self.url_processed = self.url_processed_60k if self.dataset_name == "tracking-60k" else self.url_processed_6k
        #warning = "This dataset would need ~65 GB of space after extraction. Do you want to continue? (y/n)\n"
        #if osp.exists(self.processed_paths[0]):
        #    return
        #if decide_download(self.url_processed) and input(warning).lower() == "y":
        #    print('start downloading from url')
        #    path = download_url(self.url_processed, self.root)
        #    print('start exracting zip')
        #    extract_zip(path, str(self.root) + "/processed")
        #    print('finished')
        #    os.unlink(path)
        #else:
        #    print("Stop downloading.")
        #    shutil.rmtree(self.root)
        #    exit(-1)
        pass
    def process(self):
        #all_point_clouds = os.listdir(self.raw_dir)
        all_point_clouds = self.raw_file_names

        if self.debug:
            all_point_clouds = all_point_clouds[:150]

        data_list = Parallel(n_jobs=2)(
            delayed(self.process_point_cloud)(point_cloud) for point_cloud in tqdm(all_point_clouds)
        )
        
        # Filter out None entries from data_list
        data_list = [data for data in data_list if data is not None]

        data, slices = self.collate(data_list)

        if len(os.listdir(self.raw_dir)) > 10000:
            idx_split = self.get_idx_split(data_list)
        else:
            idx_split = self.get_idx_split_old(len(data_list))
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def process_point_cloud(self, point_cloud):
        
        file_path = os.path.join(self.raw_dir, point_cloud)
    
        # Check if it's a valid file
        if not os.path.isfile(file_path):
            print(f"Skipping non-file entry: {file_path}")
            return None  # Skip this entry
        
        evtid, sector = get_event_id_sector_from_str(point_cloud)
        #data = torch.load(self.raw_dir / point_cloud)
        data = torch.load(os.path.join(self.raw_dir, point_cloud))
        df = get_dataframe(data, evtid, self.feature_names)
        
        

        eta = calc_eta(df.r, df.z)
        phi = df.phi
        data.x = (data.x / self.feature_scale).float()
        data.pos = torch.tensor([eta, phi]).T

        data.evtid = torch.tensor([evtid]).long()
        data.s = torch.tensor([sector]).long()

        # add index to the name to take care of cat_dim easily
        #data.point_pairs_index_rad = gen_point_pairs(data, k=256)
        data.knn_edge_index_k60 = to_undirected(knn_graph(data.pos, k=60, loop=True))
        #data.knn_edge_index_k60 = to_undirected(knn_graph(data.pos, k=32, loop=True))  ##just for test in 60k
        return data

    def get_idx_split(self, data_list):
        train_idx = [idx for idx, data in enumerate(data_list) if data.evtid < 29000]
        eval_idx = [idx for idx, data in enumerate(data_list) if data.evtid >= 29000]

        # half of the eval_idx is used for validation and the other half for test
        valid_idx = eval_idx[: len(eval_idx) // 2]
        test_idx = eval_idx[len(eval_idx) // 2 :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def get_idx_split_old(self, dataset_len):
        self.split = {"train": 0.7, "valid": 0.1, "test": 0.2}
        n_train = int(dataset_len * self.split["train"])
        n_train = n_train - n_train % self.n_sectors  # make sure n_train is a multiple of n_sectors
        n_valid = int(dataset_len * self.split["valid"])

        idx = np.arange(dataset_len)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def create_point_pairs_from_clusters(cluster_ids, nearby_point_pairs):
    # Get the unique cluster IDs
    unique_cluster_ids = torch.unique(cluster_ids)

    point_pairs = []
    # Iterate through each unique cluster
    for cluster_id in unique_cluster_ids:
        same_cluster_indices = (cluster_ids == cluster_id).nonzero().flatten()

        if cluster_id == 0 or same_cluster_indices.shape[0] <= 1:
            continue

        # Get indices (node ids) belonging to the same cluster
        cluster_nearby_points = nearby_point_pairs[1][torch.isin(nearby_point_pairs[0], same_cluster_indices)].unique()

        neg_pairs = torch.tensor(list(product(same_cluster_indices, cluster_nearby_points))).T
        point_pairs.append(neg_pairs)

        pos_pairs = torch.tensor(list(combinations(same_cluster_indices, 2))).T
        point_pairs.append(pos_pairs)

    point_pairs = torch.cat(point_pairs, dim=-1)
    return point_pairs


def gen_point_pairs(data, k):
    # nearby_point_pairs = to_undirected(knn_graph(data.pos, k=k, loop=False))
    nearby_point_pairs = to_undirected(radius_graph(data.pos, r=1.0, loop=False, max_num_neighbors=k))
    point_pairs = create_point_pairs_from_clusters(data.particle_id, nearby_point_pairs)
    point_pairs = remove_self_loops(to_undirected(point_pairs))[0]
    return point_pairs


def get_dataframe(evt, evtid, feature_names):
    to_df = {"evtid": evtid}
    for i, n in enumerate(feature_names):
        to_df[n] = evt.x[:, i]
    to_df["layer"] = evt.layer
    to_df["pt"] = evt.pt
    to_df["particle_id"] = evt.particle_id
    return pd.DataFrame(to_df)


def get_event_id_sector_from_str(name: str) -> tuple[int, int]:
    """
    Extracts and returns the event ID and sector ID from the file name.
    
    Args:
        name (str): The name of the file.
    
    Returns:
        tuple[int, int]: The event ID and sector ID.
    """
    # Remove any prefix that comes before "data"
    if "data" in name:
        name = name.split("data", 1)[-1]  # Get the part after "data"
        name = "data" + name  # Re-add the "data" prefix that was removed by split
    else:
        raise ValueError(f"Unexpected file name format: {name}")
    
    number_s = name.split(".")[0][len("data"):]
    evtid_s, sectorid_s = number_s.split("_s")
    evtid = int(evtid_s)
    sectorid = int(sectorid_s)
    return evtid, sectorid

def calc_eta(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Computes pseudorapidity
    (https://en.wikipedia.org/wiki/Pseudorapidity)
    """
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


if __name__ == "__main__":
    root = Path("../../data/tracking")
    parser = argparse.ArgumentParser(description="Build point clouds from raw data.")
    parser.add_argument("-d", "--dataset_name", type=str, default="tracking-6k")
    args = parser.parse_args()

    dataset = Tracking(root, args.dataset_name, debug=False)
    print(dataset)
