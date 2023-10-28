from __future__ import annotations
import torch
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from photonlib import PhotonLib
from slar.nets import SirenVis

class FlashMatchInput:
    def __init__(self):
        # input array of Flash
        self.flash_v = []
        # input array of QCluster
        self.qcluster_v = []
        # "RAW" QCluster (optional, may not be present, before x-shift)
        self.raw_qcluster_v = []
        # "RAW" flashmatch::QCluster_t (optional, may not be present, before active BB cut)
        self.all_pts_v = []
        # trajectory segment points
        self.track_v = []
        # True matches, an array of integer-pairs.
        self.true_match = []

    def shift_to_center(self,plib:PhotonLib|SirenVis):
        for q in self.qcluster_v: q.shift_to_center(plib)

    def shift_to_xmin(self,plib:PhotonLib|SirenVis):
        for q in self.qcluster_v: q.shift_to_xmin(plib)

    def shift_to_xmax(self,plib:PhotonLib|SirenVis):
        for q in self.qcluster_v: q.shift_to_xmax(plib)


class FlashMatch:
    def __init__(self, num_qclusters, num_flashes, num_pmts):
        self.loss_matrix    = torch.zeros(size=(num_qclusters, num_flashes),dtype=torch.float32)
        self.reco_x_matrix  = torch.zeros(size=(num_qclusters, num_flashes),dtype=torch.float32)
        self.duration       = torch.zeros(size=(num_qclusters, num_flashes),dtype=torch.float32)
        self.reco_pe_matrix = torch.zeros(size=(num_qclusters, num_flashes, num_pmts),dtype=torch.float32)
    
    def bipartite_match(self, loss_threshold):
        row_filter, col_filter, filtered_loss_matrix = self.filter_loss_matrix(loss_threshold)
        row_idx, col_idx = linear_sum_assignment(filtered_loss_matrix.detach().numpy())
        self.tpc_ids   = row_filter[row_idx]
        self.flash_ids = col_filter[col_idx]
        self.loss_v    = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v  = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]
        self.duration  = self.duration[self.tpc_ids, self.flash_ids]

    def local_match(self):
        """
        Reconstructs each position & p.e. to be equal to the minimum loss position & p.e. for each qcluster.
        """
        self.tpc_ids   = torch.arange(self.loss_matrix.shape[0])
        self.flash_ids = torch.argmin(self.loss_matrix, axis = 1)
        self.loss_v    = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v  = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]

    def global_match(self, loss_threshold):
        row_filter, col_filter, filtered_loss_matrix = self.filter_loss_matrix(loss_threshold)
        min_loss = np.inf

        num_tpc, num_pmt = filtered_loss_matrix.shape[0], filtered_loss_matrix.shape[1]
        col_idx = torch.arange(num_pmt)
        for row_idx in itertools.product(torch.arange(num_tpc), repeat=num_pmt):
            losses = filtered_loss_matrix[row_idx, col_idx]
            if torch.sum(losses) < min_loss:
                min_loss = torch.sum(losses)
                self.tpc_ids = row_idx
                self.flash_ids = col_idx

        self.tpc_ids   = row_filter[self.tpc_ids]
        self.flash_ids = col_filter[self.flash_ids]

        self.loss_v    = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v  = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]

    def filter_loss_matrix(self, loss_threshold):
        """
        Filters the loss matrix by keeping all rows and columns that has at least
        one element less than the loss threshold.

        Args:
        ---
        loss_threshold (float): The minimum loss value to keep a row or column.

        Returns:
        ---
        Tuple of three numpy arrays:
            * row_filter: The indices of the rows to keep.
            * col_filter: The indices of the columns to keep.
            * filtered_loss_matrix: The filtered loss matrix.
        """
        row_filter = []
        col_filter = []
        for i, row in enumerate(self.loss_matrix):
            if torch.min(row) <= loss_threshold:
                row_filter.append(i)

        for j in range(self.loss_matrix.shape[1]):
            col = self.loss_matrix[:, j]
            if torch.min(col) <= loss_threshold:
                col_filter.append(j)


        return torch.as_tensor(row_filter), torch.as_tensor(col_filter), self.loss_matrix[row_filter, :][:, col_filter]