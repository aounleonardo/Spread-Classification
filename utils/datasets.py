from itertools import groupby
import math
from operator import itemgetter
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional

from torch.utils.data import Dataset, Subset

from spread_classification.utils import load_graph


class _Classification_Dataset(Dataset):
    def __init__(self, datapath: str, extension: str):
        self.extension = extension
        self.datapath = datapath

        self.ratings_map, self.labels_map = self._get_ratings_and_labels_maps(datapath)

        self.labels = {
            self._extract_id(name): label
            for rating, label in self.ratings_map.items()
            for name in os.listdir(os.path.join(self.datapath, rating))
            if name.endswith(self.extension)
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id_):
        raise NotImplementedError(f"{type(self)} is an Abstract Class")

    def get_subsets(self, lengths: List[int], label_ratios: Optional[List[int]]):
        if label_ratios is None:
            label_ratios = list(self.get_label_ratios().values())
        if sum(label_ratios) != 1.0:
            raise ValueError(f"Sum of ratios in {type(self)}.get_subsets must be 1.0")
        label_groups = self._get_label_groups()
        sum_lengths = sum(lengths)
        for (label, group), ratio in zip(label_groups.items(), label_ratios):
            if len(group) < int(sum_lengths * ratio):
                raise ValueError(
                    f"{self} cannot provide {int(sum_lengths * ratio)} samples of label {self.labels_map[label]}"
                )
        groups = {
            label: random.sample(group, len(group))
            for label, group in label_groups.items()
        }

        subsets = []
        start = 0
        for length in lengths:
            keys = []
            for group, ratio in zip(groups.values(), label_ratios):
                group_start = math.floor(ratio * start)
                group_add = math.floor(ratio * length)
                keys += group[group_start : group_start + group_add]
            subsets.append(Subset(self, random.sample(keys, len(keys))))
            start += length

        return subsets

    def _extract_id(self, path):
        if self.extension not in path:
            return None
        return Path(path).stem

    def _get_dir(self, id_) -> str:
        label = self.labels[id_]
        return os.path.join(self.datapath, self.labels_map[label])

    def _get_label_groups(self):
        return {
            label: [id_ for id_, label in group]
            for label, group in groupby(
                sorted(self.labels.items(), key=itemgetter(1)), itemgetter(1)
            )
        }

    def get_label_ratios(self):
        label_groups = self._get_label_groups()
        return {label: len(ids) / len(self) for (label, ids) in label_groups.items()}

    def _get_ratings_and_labels_maps(
        self, datapath: str
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        ratings = os.listdir(datapath)
        ratings_map = {rating: index for index, rating in enumerate(sorted(ratings))}
        labels_map = {label: rating for rating, label in ratings_map.items()}
        return ratings_map, labels_map


class GraphsDataset(_Classification_Dataset):
    def __init__(self, datapath: str):
        super().__init__(datapath, ".bin")

    def __getitem__(self, id_):
        if id_ not in self.labels:
            raise KeyError()
        graph = load_graph(os.path.join(self._get_dir(id_), f"{id_}{self.extension}"))
        return graph, self.labels[id_]


class StoriesDataset(_Classification_Dataset):
    def __init__(self, datapath: str):
        super().__init__(datapath, ".txt")

    def __getitem__(self, id_):
        if id_ not in self.labels:
            raise KeyError()
        with open(os.path.join(self._get_dir(id_), f"{id_}{self.extension}")) as file:
            return file.read(), self.labels[id_]


class UrlsDataset(_Classification_Dataset):
    def __init__(self, datapath: str):
        super().__init__(datapath, "")

        self.labels = {
            id_: label
            for (id_, label) in self.labels.items()
            if os.path.exists(self[id_])
        }

    def __getitem__(self, id_):
        if id_ not in self.labels:
            raise KeyError()
        return os.path.join(self._get_dir(id_), id_, "tweets")

    def sizeof(self, id_):
        return len(os.listdir(self[id_]))
