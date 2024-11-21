
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
modifications of the matcher from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class HungarianMatcher_Cells():
    """
    This class computes an assignment between the targets and the predictions of the network
    """

    def match(self, outputs, targets, maxDist = 5):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains:
                 "points": Tensor of dim [num_queries, 2] with the predicted point coordinates

            targets: This is a dict containing:
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        num_queries = outputs["points"].shape[0]

        out_points = outputs["points"]

        # Also concat the target points
        tgt_points = targets["points"]

        # Compute the L2 cost between points. 
        cost_point = cdist(out_points, tgt_points)

        # replace distant points with large value toa avoid matching
        cost_point[cost_point>maxDist] = 1e3
        
        # match
        indices_i,indices_j = linear_sum_assignment(cost_point) 

        # Keep true matches (distance <=maxDist)
        true_matches = []
        for cnt, (i, j) in enumerate(zip(indices_i, indices_j)):
            if cost_point[i,j]<=maxDist:
                true_matches.append(cnt)

        return [indices_i[true_matches], indices_j[true_matches]], cost_point 



def build_matcher_cells():
    return HungarianMatcher_Cells()