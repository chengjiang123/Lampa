from torch_geometric.loader import DataLoader
from datasets import Tracking, TrackingTransform


from torch_geometric.loader import DenseDataLoader

def get_data_loader_new(dataset, idx_split, batch_size):
    return {
      'train': DenseDataLoader(dataset[idx_split['train']],
                               batch_size=batch_size, shuffle=True),
      'valid': DenseDataLoader(dataset[idx_split['valid']],
                               batch_size=batch_size, shuffle=False),
      'test' : DenseDataLoader(dataset[idx_split['test']],
                               batch_size=batch_size, shuffle=False),
    }

def get_data_loader(dataset, idx_split, batch_size):
    train_loader = DataLoader(
        dataset[idx_split["train"]],
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset[idx_split["valid"]],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset[idx_split["test"]],
        batch_size=batch_size,
        shuffle=False,
    )
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def get_dataset(dataset_name, data_dir):
    print(dataset_name)
    if dataset_name in ["tracking-6k","tracking-3k","tracking-10k","tracking-15k","tracking-20k", "tracking-60k"]:
        dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name)
    #elif "segtracking" in dataset_name:
    #    dataset = SegTracking(data_dir, transform=SegTrackingTransform(), dataset_name=dataset_name)
    #elif "regression" in dataset_name:
    #    dataset = Regression(data_dir, transform=RegressionTransform(), dataset_name=dataset_name)
    #elif dataset_name == "pileup":
    #    dataset = Pileup(data_dir, transform=PileupTransform())
    else:
        raise NotImplementedError
    dataset.dataset_name = dataset_name
    print(dataset)
    print(dataset.dataset_name)
    return dataset
