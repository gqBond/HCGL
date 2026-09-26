import torch
import torch.nn as nn
import torch.nn.functional as F


class my_sampler(nn.Module):
    """DMSG diversity sampler for task-local node representations."""

    def __init__(self, args):
        super().__init__()

    def forward(self, ids_per_cls_train, budget, representations):
        if not ids_per_cls_train:
            return []

        budget = int(budget)
        representations = representations[:, :len(ids_per_cls_train)]
        representations = F.softmax(representations, dim=1)

        all_ids = []
        class_offsets = [0]
        for ids in ids_per_cls_train:
            all_ids.extend(int(node_id) for node_id in ids)
            class_offsets.append(class_offsets[-1] + len(ids))

        num_classes = len(ids_per_cls_train)
        selected = [[] for _ in range(num_classes)]
        for class_id in range(num_classes):
            low = class_offsets[class_id]
            high = class_offsets[class_id + 1]
            if low == high:
                continue
            first = int(representations[low:high, class_id].argmax().item()) + low
            selected[class_id].append(first)

        min_distances = torch.full(
            (num_classes, representations.shape[0]),
            1000.0,
            device=representations.device,
        )
        for class_id, class_selected in enumerate(selected):
            if not class_selected:
                continue
            first = class_selected[0]
            min_distances[class_id] = torch.minimum(
                min_distances[class_id],
                self.euclidean_distance(representations[first], representations),
            )
            min_distances[class_id, first] = -1000

        for _ in range(1, budget):
            for class_id, class_ids in enumerate(ids_per_cls_train):
                if len(selected[class_id]) >= min(len(class_ids), budget):
                    continue
                low = class_offsets[class_id]
                high = class_offsets[class_id + 1]
                distance = min_distances[class_id] + 2 * (
                    min_distances.sum(dim=0) - min_distances[class_id]
                )
                distance[min_distances[class_id] < -99] = -1000
                next_id = int(distance[low:high].argmax().item()) + low
                selected[class_id].append(next_id)
                min_distances[class_id] = torch.minimum(
                    min_distances[class_id],
                    self.euclidean_distance(
                        representations[next_id], representations
                    ),
                )
                selected_tensor = torch.tensor(
                    selected[class_id],
                    dtype=torch.long,
                    device=representations.device,
                )
                min_distances[class_id, selected_tensor] = -1000

        return [all_ids[index] for class_ids in selected for index in class_ids]

    @staticmethod
    def euclidean_distance(vector, matrix):
        return torch.cdist(vector.unsqueeze(0), matrix, p=2).squeeze(0)
