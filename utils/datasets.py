from itertools import groupby
import math
from operator import itemgetter
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset, Subset

from spread_classification.utils import load_graph


class GraphDataset(Dataset):
    def __init__(self, datapath: str, extension: str = ".bin"):
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
        if id_ not in self.labels:
            raise KeyError()
        label = self.labels[id_]
        dir_ = os.path.join(self.datapath, self.labels_map[label])
        graph = load_graph(os.path.join(dir_, f"{id_}{self.extension}"))
        return graph, label

    def get_subsets(self, lengths: List[int], preserve_ratios: bool = True):
        if sum(lengths) > len(self):
            raise ValueError(
                f"{self} of len {len(self)} cannot provide {sum(lengths)} samples"
            )
        groups = {
            label: random.sample(group, len(group))
            for label, group in self._get_label_groups().items()
        }

        subsets = []
        start = 0
        ratio = 1.0 / len(groups)
        for length in lengths:
            keys = []
            for group in groups.values():
                if preserve_ratios:
                    ratio = len(group) / len(self)
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

    def _get_label_groups(self):
        return {
            label: [id_ for id_, label in group]
            for label, group in groupby(
                sorted(self.labels.items(), key=itemgetter(1)), itemgetter(1)
            )
        }

    def _get_ratings_and_labels_maps(
        self, datapath: str
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        ratings = os.listdir(datapath)
        ratings_map = {rating: index for index, rating in enumerate(sorted(ratings))}
        labels_map = {label: rating for rating, label in ratings_map.items()}
        return ratings_map, labels_map
