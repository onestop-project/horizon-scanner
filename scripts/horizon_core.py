"""Shared code of the IAS horizon scanner.

Imported by ``05_model_training.ipynb`` (training and the first export)
and ``06_static_scanner.ipynb`` (lists from the saved checkpoint, without
training), so the model classes, the scanner and the export code exist
once. Keep this file in the same folder as the notebooks.

Contents
--------
1. Model architecture: ``CountryAutoencoder``, ``RelationalEcologicalGNN``,
   ``TradePathwayEncoder``, ``InvasionRiskModel``, ``PairFeatureBuilder``,
   the scoring helpers, the positive-unlabeled loss (``pu_risk``) and the
   self-supervised pre-training of the graph encoder
   (``pretrain_on_links``).
2. ``Scanner``: ranking, uncertainty, feature contributions and GNN
   saliency maps for one location.
3. Impact data: kept out of the lists. Notebook 07 writes
   ``species_impacts_master.csv``; the viewer joins it on ``gbif_id``.
4. ``ChecklistChecks``: native-range and GRIIS checklist checks that are
   applied to a ranked list before it is exported.
5. ``load_static_scanner`` and ``export_scans``: re-load a checkpoint and
   write the files the viewer reads.

Requires: torch, torch_geometric, numpy, pandas, tqdm (matplotlib only for
``plot_saliency_map``; pygbif only for the live GBIF check;
country_converter only when the GRIIS file has no ISO3 column).
"""
import json
import math
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv
from tqdm.auto import tqdm

# Checkpoint formats this file can read (written by notebook 05).
SUPPORTED_CHECKPOINT_FORMATS = (4, 5, 6)


# ======================================================================
# 0. Small helpers
# ======================================================================
def check(condition, message):
    """Raise a ValueError with ``message`` when ``condition`` is false."""
    if not condition:
        raise ValueError(message)


def slugify(text):
    """Lower-case, underscore-separated version of ``text``."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def pretty_feature(column):
    """Readable label for a feature column (used in the viewer)."""
    label = re.sub(r"^log1p_", "", column)
    label = re.sub(r"^wc_bio\d\d_", "", label)
    label = re.sub(r"^wc_", "", label)
    return label.replace("_", " ")


# ======================================================================
# 1. Model architecture
# ======================================================================
LEAF_NAMES = ["sp_emb", "co_latent", "co_graph", "niche_mean", "niche_std",
              "target_climate", "koppen", "proximity", "context",
              "trade_import", "trade_export", "trade_source", "biotic"]
PAIR_LEAVES = [n for n in LEAF_NAMES if n != "sp_emb"]
TAXONOMY_PREFIXES = ("belongs_to_", "has_")  # edges kept by DropEdge
GNN_CONVS = ("sage", "gatv2")


class CountryAutoencoder(nn.Module):
    """Compress the full location feature vector into a latent code."""

    def __init__(self, n_features, latent_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim), nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        """Return (latent code, reconstruction)."""
        z = self.encoder(x)
        return z, self.decoder(z)

    def encode(self, x):
        """Latent code only."""
        return self.encoder(x)


class RelationalEcologicalGNN(nn.Module):
    """Two-layer heterogeneous graph encoder over species and taxonomy.

    Layer 1 aggregates direct neighbours, layer 2 two-hop structure. Each
    layer has its own convolution per edge type (relation). Node types
    without incoming messages keep their previous state.

    ``conv``
        ``"gatv2"``: attention per relation (GATv2; Brody, Alon & Yahav
        2022). Each node weighs its neighbours, and the edge weight
        (scaled log number of GloBI records behind the edge) enters the
        attention as an edge feature, so well-supported interactions can
        count for more than single records.
        ``"sage"``: unweighted mean over the neighbours (GraphSAGE).
    ``in_dims`` / ``num_nodes``
        Given for taxonomy node types (genus ... kingdom), they switch on
        a *learned* input for those nodes: a linear projection of the
        node's input features plus one learned vector per node (zero at
        the start). Without them, taxonomy inputs must already have
        ``dim`` columns and are used as they are.
    ``edge_dropout``
        In training mode, the share of interaction edges removed per
        forward pass (DropEdge); taxonomy edges are always kept. The same
        reduced graph is used by both layers.

    Edge weights are not part of the saved weights: set
    ``encoder.edge_weight = {edge_type: tensor}`` after construction (or
    pass ``edge_weight_dict`` to ``forward``). Edge types without a
    weight get 1.
    """

    def __init__(self, dim, edge_types, node_types, n_sp_features,
                 dropout=0.4, feature_dropout=0.35, edge_dropout=0.0,
                 conv="sage", heads=2, in_dims=None, num_nodes=None):
        super().__init__()
        check(conv in GNN_CONVS, f"conv must be one of {GNN_CONVS}")
        check(conv != "gatv2" or dim % heads == 0,
              "The embedding size must be divisible by the number of "
              "attention heads.")
        self.conv_type = conv
        self.sp_projector = nn.Linear(n_sp_features, dim)
        in_dims, num_nodes = in_dims or {}, num_nodes or {}
        taxa = [nt for nt in node_types if nt != "species"]
        self.tax_projector = nn.ModuleDict(
            {nt: nn.Linear(in_dims[nt], dim) for nt in taxa
             if nt in in_dims})
        self.node_embedding = nn.ParameterDict(
            {nt: nn.Parameter(torch.zeros(num_nodes[nt], dim))
             for nt in taxa if nt in num_nodes})
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = float(edge_dropout)
        self.droppable = [et for et in edge_types
                          if not et[1].startswith(TAXONOMY_PREFIXES)]
        self.edge_weight = {}

        def layer(et):
            if conv == "gatv2":
                return GATv2Conv((dim, dim), dim // heads, heads=heads,
                                 edge_dim=1, add_self_loops=False)
            return (SAGEConv(dim, dim) if et[0] == et[2]
                    else SAGEConv((dim, dim), dim))

        def schema():
            return {et: layer(et) for et in edge_types}

        self.conv1 = HeteroConv(schema(), aggr="mean")
        self.conv2 = HeteroConv(schema(), aggr="mean")
        self.norm1 = nn.ModuleDict({n: nn.LayerNorm(dim) for n in node_types})
        self.norm2 = nn.ModuleDict({n: nn.LayerNorm(dim) for n in node_types})
        # GATv2 without self-loops has no term for the node itself, so a
        # node would lose its own features; GraphSAGE has one built in.
        self.self1 = nn.ModuleDict()
        self.self2 = nn.ModuleDict()
        if conv == "gatv2":
            self.self1.update({n: nn.Linear(dim, dim) for n in node_types})
            self.self2.update({n: nn.Linear(dim, dim) for n in node_types})

    def drop_edges(self, edge_index_dict, weights):
        """Random subset of the interaction edges (training mode only).

        Returns the reduced edge dict and the matching edge weights.
        """
        if not (self.training and self.edge_dropout > 0):
            return edge_index_dict, weights
        edges, kept = dict(edge_index_dict), dict(weights)
        for et in self.droppable:
            if et in edges:
                ei = edges[et]
                keep = (torch.rand(ei.shape[1], device=ei.device)
                        >= self.edge_dropout)
                edges[et] = ei[:, keep]
                w = kept.get(et)
                if w is not None and len(w) == len(keep):
                    kept[et] = w[keep]
        return edges, kept

    def edge_attributes(self, edges, weights):
        """(E, 1) edge feature per edge type; 1 where no weight is known."""
        out = {}
        for et, ei in edges.items():
            w = weights.get(et)
            if w is None or len(w) != ei.shape[1]:
                w = torch.ones(ei.shape[1], device=ei.device)
            out[et] = w.to(ei.device).float().view(-1, 1)
        return out

    def inputs(self, x_dict):
        """Project the node features of every type to the embedding size."""
        x = {}
        for nt, feats in x_dict.items():
            if nt == "species":
                h = self.sp_projector(feats)
            elif nt in self.tax_projector:
                h = self.tax_projector[nt](feats)
            else:
                h = feats
            if nt in self.node_embedding:
                h = h + self.node_embedding[nt]
            x[nt] = h
        return x

    def propagate(self, conv, own, x, edges, weights):
        """One round of message passing with the configured convolution.

        ``own`` holds the per-type layers that add a node's own state to
        the messages it receives (attention only).
        """
        if self.conv_type != "gatv2":
            return conv(x, edges)
        out = conv(x, edges,
                   edge_attr_dict=self.edge_attributes(edges, weights))
        return {k: v + own[k](x[k]) for k, v in out.items()}

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        """Return a dict of node embeddings per node type.

        ``edge_weight_dict`` replaces the stored ``edge_weight`` for this
        call (used when only part of the edges is passed in).
        """
        weights = (self.edge_weight if edge_weight_dict is None
                   else edge_weight_dict)
        edges, weights = self.drop_edges(edge_index_dict, weights or {})
        x = self.inputs(x_dict)
        if self.training:
            x = {k: self.feature_dropout(v) for k, v in x.items()}
        h = self.propagate(self.conv1, self.self1, x, edges, weights)
        h = {k: (self.norm1[k](self.dropout(F.gelu(h[k]))) if k in h
                 else x[k]) for k in x}
        h2 = self.propagate(self.conv2, self.self2, h, edges, weights)
        return {k: (self.norm2[k](F.gelu(h2[k])) if k in h2 else h[k])
                for k in h}


class TradePathwayEncoder(nn.Module):
    """Fuse import, routed export and source exporter strength.

    Channel attention weights the import x export interaction per
    channel. With ``sp_dim > 0`` the attention also sees a compressed
    species embedding, so the weight of a pathway depends on the species:
    used tyres can matter for a mosquito and not for a tree. Without it
    the attention depends on the trade vectors only. ``src_dim = 0``
    drops the exporter-strength input.
    """

    def __init__(self, num_channels, latent_dim, hidden_dim=32, src_dim=0,
                 sp_dim=0, sp_hidden=16):
        super().__init__()
        self.src_dim = src_dim
        self.sp_dim = sp_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_channels * 3 + src_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim), nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        if sp_dim > 0:
            self.species_context = nn.Sequential(
                nn.Linear(sp_dim, sp_hidden), nn.GELU())
        self.channel_attn = nn.Sequential(
            nn.Linear(num_channels * 2 + (sp_hidden if sp_dim > 0 else 0),
                      num_channels), nn.Sigmoid())

    def forward(self, import_vec, export_vec, src_vec, sp_emb=None):
        """Return the trade latent vector per pair."""
        attn_in = [import_vec, export_vec]
        if self.sp_dim > 0:
            attn_in.append(self.species_context(sp_emb))
        attn = self.channel_attn(torch.cat(attn_in, dim=-1))
        parts = [import_vec, export_vec, import_vec * export_vec * attn]
        if self.src_dim > 0:
            parts.append(src_vec)
        return self.encoder(torch.cat(parts, dim=-1))


class InvasionRiskModel(nn.Module):
    """Seven-stream classifier over the per-pair leaves.

    ``dims`` keys: sp_dim, co_latent_dim, n_niche, n_koppen, n_proximity,
    n_context, n_biotic, n_channels, src_dim, trade_latent_dim,
    topo_dropout, climate_gate ("none", "soft" or "learned"),
    gate_floor; optional: n_co_graph (size of the location-graph leaf,
    0 = none) and trade_species_attention (channel weights of the trade
    stream depend on the species embedding). Stored in the checkpoint to
    rebuild the model.
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dict(dims)
        d = self.dims
        sp, nn_ = d["sp_dim"], d["n_niche"]
        self.topo_dropout = d.get("topo_dropout", 0.0)
        self.n_co_graph = int(d.get("n_co_graph", 0))
        self.trade_by_species = bool(d.get("trade_species_attention", False))
        self.trade_encoder = TradePathwayEncoder(
            d["n_channels"], d["trade_latent_dim"], 32, d["src_dim"],
            sp_dim=sp if self.trade_by_species else 0)
        self.topo_stream = nn.Sequential(
            nn.Linear(sp + d["co_latent_dim"] + self.n_co_graph, sp // 2),
            nn.GELU(),
            nn.Dropout(0.4), nn.Linear(sp // 2, sp // 4), nn.GELU())
        self.niche_stream = nn.Sequential(
            nn.Linear(nn_ * 3 + d["n_koppen"], sp), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(sp, sp // 2), nn.GELU())
        self.suitability_stream = nn.Sequential(
            nn.Linear(nn_ * 2, nn_ * 4), nn.GELU(),
            nn.Linear(nn_ * 4, nn_), nn.Sigmoid())
        self.trade_proj = nn.Sequential(
            nn.Linear(d["trade_latent_dim"], sp // 4), nn.GELU())
        self.bio_proj = nn.Sequential(
            nn.Linear(d["n_biotic"], 16), nn.GELU(),
            nn.Linear(16, 8), nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Linear(d["n_context"], 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 8), nn.GELU())
        self.proximity_proj = nn.Sequential(
            nn.Linear(d["n_proximity"], 8), nn.GELU(),
            nn.Linear(8, 8), nn.GELU())
        self.gate_mode = d.get("climate_gate", "none")
        self.gate_floor = float(d.get("gate_floor", 0.25))
        # sigma of the Gaussian suitability; set per training (Step 10)
        self.register_buffer("gate_sigma", torch.tensor(1.0))
        if self.gate_mode == "learned":
            self.climate_gate = nn.Sequential(
                nn.Linear(1 + nn_, nn_ * 2), nn.GELU(),
                nn.Linear(nn_ * 2, 1), nn.Sigmoid())
        combined = sp // 4 + sp // 2 + nn_ + sp // 4 + 8 + 8 + 8
        self.classifier = nn.Sequential(
            nn.Linear(combined, combined // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(combined // 2, 1))

    @staticmethod
    def niche_distance(niche_mean, niche_std, target):
        """Per-axis z-difference and Mahalanobis (diagonal) distance."""
        z = (niche_mean - target) / niche_std
        return z, z.pow(2).sum(dim=-1, keepdim=True).sqrt()

    def gate(self, niche_mean, niche_std, target):
        """Climate suitability in [0, 1] from the niche distance.

        "learned": the gate network. "none" and "soft": the closed
        form exp(-d^2 / 2 sigma^2) ("none": a diagnostic only).
        """
        z, m = self.niche_distance(niche_mean, niche_std, target)
        if self.gate_mode == "learned":
            return self.climate_gate(torch.cat([m, z.pow(2)], dim=-1))
        return torch.exp(-m.pow(2) / (2 * self.gate_sigma.pow(2)))

    def forward(self, x, detach_gate=True):
        """Return (logits, climate suitability) for a dict of leaves.

        Leaves: see LEAF_NAMES. With a learned gate,
        ``detach_gate=False`` lets gradients flow through it; used only
        by the explanations. The soft gate is a fixed function of the
        climate inputs and is never detached. Without a gate the
        suitability is returned for diagnostics only.
        """
        z, _ = self.niche_distance(x["niche_mean"], x["niche_std"],
                                   x["target_climate"])
        gate = self.gate(x["niche_mean"], x["niche_std"],
                         x["target_climate"])
        topo_in = [x["sp_emb"], x["co_latent"]]
        if self.n_co_graph:
            topo_in.append(x["co_graph"])
        topo = self.topo_stream(torch.cat(topo_in, dim=-1))
        scale = None  # soft gate: applied to A, D, E and G
        if self.gate_mode == "learned":
            topo = topo * (gate.detach() if detach_gate else gate)
        elif self.gate_mode == "soft":
            scale = self.gate_floor + (1.0 - self.gate_floor) * gate
            topo = topo * scale
        if self.training and self.topo_dropout > 0:
            keep = (torch.rand(topo.shape[0], 1, device=topo.device)
                    > self.topo_dropout).float()
            topo = topo * keep
        niche = self.niche_stream(torch.cat(
            [x["niche_mean"], x["target_climate"], z.abs(), x["koppen"]],
            dim=-1))
        suit = self.suitability_stream(torch.cat(
            [x["niche_mean"], x["target_climate"]], dim=-1))
        trade = self.trade_proj(self.trade_encoder(
            x["trade_import"], x["trade_export"], x["trade_source"],
            x["sp_emb"] if self.trade_by_species else None))
        bio = self.bio_proj(x["biotic"])
        ctx = self.context_proj(x["context"])
        prox = self.proximity_proj(x["proximity"])
        if scale is not None:
            trade, bio, prox = trade * scale, bio * scale, prox * scale
        logits = self.classifier(torch.cat(
            [topo, niche, suit, trade, bio, ctx, prox], dim=-1)).squeeze(-1)
        return logits, gate.squeeze(-1)


class PairFeatureBuilder:
    """Build the per-pair leaves from static tensors and a range matrix.

    ``static`` keys: co_niche, co_koppen, co_koppen_available, co_context,
    co_latent, dist_km, same_region, trade, co_import, co_export, pos_opp,
    neg_opp, pos_prior, neg_prior, n_pos, n_neg, sp_bio_static,
    n_locations, proximity_scale_km, neighbour_radius_km, niche_std_floor,
    niche_prior_strength, niche_prior_min_locations,
    biotic_prior_strength, thin_prob, thin_keep_min. Optional (location
    graph): co_graph (N x k, neighbourhood profile of every location) and
    biota_sim (N x N, similarity of the native biota of two locations).

    ``range_mat`` (S x N) marks each species' known range for this split.
    ``bio_neutral`` (S x 4) is the effort-neutral species part of the
    biotic leaf, used by ``build(..., neutral=True)``.
    """

    def __init__(self, static, range_mat, src_aggs=("mean", "max"),
                 chunk=4096, bio_neutral=None):
        self.s = static
        self.src_aggs = tuple(src_aggs)
        self.chunk = chunk
        self.P = range_mat.float()
        self.bio_neutral = (static["sp_bio_static"] if bio_neutral is None
                            else bio_neutral)
        clim = static["co_niche"]
        self.g_mean = clim.mean(dim=0, keepdim=True)
        self.g_var = clim.var(dim=0, unbiased=False,
                              keepdim=True).clamp(min=1e-8)
        self.prior_var = self._typical_niche_variance(clim)

    def _typical_niche_variance(self, clim):
        """Median per-axis niche variance of species with a large range."""
        n = self.P.sum(dim=1)
        big = n >= self.s["niche_prior_min_locations"]
        if int(big.sum()) < 10:  # tiny data sets: relax the threshold
            big = n >= 3
        if int(big.sum()) == 0:
            return self.g_var.clone()
        rows, nb = self.P[big], n[big].unsqueeze(1)
        mean = (rows @ clim) / nb
        var = ((rows @ clim.pow(2)) / nb - mean.pow(2)).clamp(min=0)
        return var.median(dim=0, keepdim=True).values.clamp(min=1e-6)

    def range_mask(self, sp, co, loo=True):
        """(b, N) copy of the range rows, target left out if ``loo``."""
        mask = self.P[sp].clone()
        if loo:
            mask[torch.arange(len(co), device=mask.device), co] = 0.0
        return mask

    def thin(self, mask):
        """Drop a random share of range locations (training augmentation).

        A share ``thin_prob`` of the rows keeps each location with a
        probability drawn from [thin_keep_min, 1]; a non-empty range keeps
        at least one location.
        """
        prob, keep_min = self.s["thin_prob"], self.s["thin_keep_min"]
        if prob <= 0:
            return mask
        b, dev = mask.shape[0], mask.device
        applied = torch.rand(b, 1, device=dev) < prob
        frac = keep_min + (1.0 - keep_min) * torch.rand(b, 1, device=dev)
        r = torch.rand_like(mask)
        out = mask * ((r < frac) | ~applied).float()
        lost = (out.sum(dim=1) == 0) & (mask.sum(dim=1) > 0)
        if bool(lost.any()):
            rows = lost.nonzero(as_tuple=True)[0]
            cols = ((r[rows] + 1e-6) * mask[rows]).argmax(dim=1)
            out[rows, cols] = 1.0
        return out

    def niche(self, sp, co, loo=True, mask=None):
        """Realised climate niche (mean, SD) over the range, shrunk.

        SD^2 = (SS + k * typical variance) / (n - 1 + k); an empty range
        gets the global mean and spread (see the markdown above).
        """
        if mask is None:
            mask = self.range_mask(sp, co, loo)
        clim = self.s["co_niche"]
        n = mask.sum(dim=1, keepdim=True)
        s, q = mask @ clim, mask @ clim.pow(2)
        has = n > 0.5
        mean = torch.where(has, s / n.clamp(min=1), self.g_mean)
        ss = (q - n * mean.pow(2)).clamp(min=0)
        k = self.s["niche_prior_strength"]
        var = torch.where(has, (ss + k * self.prior_var)
                          / ((n - 1).clamp(min=0) + k), self.g_var)
        return mean, var.sqrt().clamp(min=self.s["niche_std_floor"])

    def koppen(self, sp, co, loo=True, mask=None):
        """[profile overlap, class coverage, data available], (b, 3).

        overlap  = sum over classes of min(range profile, target share)
        coverage = target area share in classes seen anywhere in the range
        Range locations without Koppen-Geiger data are left out of the
        profile; without data on either side all three are 0.
        """
        if mask is None:
            mask = self.range_mask(sp, co, loo)
        kg = self.s["co_koppen"]
        avail = self.s["co_koppen_available"].float()
        tgt = kg[co]
        n_kg = mask @ avail
        profile = (mask @ kg) / n_kg.clamp(min=1).unsqueeze(1)
        seen = ((mask @ (kg > 0).float()) > 0.5).float()
        has = ((n_kg > 0.5) & (avail[co] > 0.5)).float()
        overlap = torch.minimum(profile, tgt).sum(dim=1) * has
        coverage = (tgt * seen).sum(dim=1) * has
        return torch.stack([overlap, coverage, has], dim=1)

    def proximity(self, sp, co, loo=True, mask=None):
        """[closeness of nearest range location, log #range locations
        within the radius (scaled), share of range in the same region,
        log range size (scaled)], (b, 4).

        With ``biota_sim`` in the static inputs two more columns follow:
        the mean and the largest similarity between the native biota of
        the target and that of the range locations (b, 6). The range has
        the target left out, like every other range feature.
        """
        if mask is None:
            mask = self.range_mask(sp, co, loo)
        d = self.s["dist_km"][co]
        nearest = torch.where(mask > 0, d, torch.full_like(d, float("inf")))
        closeness = torch.exp(-nearest.amin(dim=1)
                              / self.s["proximity_scale_km"])
        near = (mask * (d <= self.s["neighbour_radius_km"]).float()).sum(1)
        size = mask.sum(dim=1)
        same = ((mask * self.s["same_region"][co].float()).sum(1)
                / size.clamp(min=1))
        scale = math.log1p(self.s["n_locations"])
        cols = [closeness, torch.log1p(near) / scale, same,
                torch.log1p(size) / scale]
        if "biota_sim" in self.s:
            sim = mask * self.s["biota_sim"][co]
            cols += [sim.sum(dim=1) / size.clamp(min=1), sim.amax(dim=1)]
        return torch.stack(cols, dim=1)

    def trade(self, sp, co, loo=True, mask=None):
        """Import strength of the target, trade routed into the target from
        the range, and exporter strength of the range per aggregation."""
        if mask is None:
            mask = self.range_mask(sp, co, loo)
        n_src = mask.sum(dim=1, keepdim=True).clamp(min=1)
        to_dest = self.s["trade"][:, :, co].permute(2, 0, 1)  # (b, C, N)
        export = (to_dest * mask.unsqueeze(1)).sum(dim=2) / n_src
        exporter = self.s["co_export"]
        cols = []
        for agg in self.src_aggs:
            if agg == "mean":
                cols.append((mask @ exporter) / n_src)
            elif agg == "max":
                cols.append((mask.unsqueeze(2) * exporter.unsqueeze(0))
                            .amax(dim=1))
            else:
                raise ValueError(f"Unknown exporter-strength agg: {agg}")
        src = (torch.cat(cols, dim=1) if cols
               else mask.new_zeros(len(sp), 0))
        return self.s["co_import"][co], export, src

    def biotic(self, sp, co, neutral=False):
        """[pos. opportunity, neg. opportunity, has pos, has neg,
        log #pos, log #neg], (b, 6). Opportunities are shrunk (Step 5)."""
        opp = torch.stack([self.s["pos_opp"][sp, co],
                           self.s["neg_opp"][sp, co]], dim=1)
        species_part = (self.bio_neutral if neutral
                        else self.s["sp_bio_static"])[sp]
        return torch.cat([opp, species_part], dim=1)

    def raw_opportunity(self, sign, sp, co):
        """Observed (unshrunk) partner share, for the exports."""
        shrunk = self.s[f"{sign}_opp"][sp, co]
        n = self.s[f"n_{sign}"][sp]
        k = self.s["biotic_prior_strength"]
        prior = self.s[f"{sign}_prior"][co]
        raw = ((n + k) * shrunk - k * prior) / n.clamp(min=1)
        return torch.where(n > 0, raw, torch.zeros_like(raw)).clamp(0, 1)

    @torch.no_grad()
    def build(self, sp, co, loo=True, neutral=False, thin=False):
        """All leaves except ``sp_emb`` for index tensors ``sp``, ``co``.

        ``neutral``: effort-neutral species part of the biotic leaf.
        ``thin``: range-thinning augmentation (training pairs only).
        """
        if len(sp) == 0:
            raise ValueError("build() called with no pairs")
        parts = defaultdict(list)
        for i in range(0, len(sp), self.chunk):
            s, c = sp[i:i + self.chunk], co[i:i + self.chunk]
            mask = self.range_mask(s, c, loo)
            if thin:
                mask = self.thin(mask)
            mean, std = self.niche(s, c, mask=mask)
            imp, exp, src = self.trade(s, c, mask=mask)
            values = {
                "co_latent": self.s["co_latent"][c],
                "niche_mean": mean, "niche_std": std,
                "target_climate": self.s["co_niche"][c],
                "koppen": self.koppen(s, c, mask=mask),
                "proximity": self.proximity(s, c, mask=mask),
                "context": self.s["co_context"][c],
                "trade_import": imp, "trade_export": exp,
                "trade_source": src,
                "biotic": self.biotic(s, c, neutral),
            }
            if "co_graph" in self.s:
                values["co_graph"] = self.s["co_graph"][c]
            for key, value in values.items():
                parts[key].append(value)
        return {k: torch.cat(v) for k, v in parts.items()}

    @torch.no_grad()
    def describe(self, sp, co):
        """Readable raw values per pair (for the export, not the model)."""
        mask = self.range_mask(sp, co, loo=True)
        mean, std = self.niche(sp, co, mask=mask)
        _, mdist = InvasionRiskModel.niche_distance(
            mean, std, self.s["co_niche"][co])
        d = self.s["dist_km"][co]
        nearest = torch.where(mask > 0, d, torch.full_like(d, float("inf")))
        return {
            "feat_mahalanobis_distance": mdist.squeeze(1),
            "feat_koppen_overlap": self.koppen(sp, co, mask=mask)[:, 0],
            "feat_nearest_range_km": nearest.amin(dim=1),
            "feat_n_range_locations": mask.sum(dim=1),
            "feat_biotic_opp_positive": self.raw_opportunity("pos", sp, co),
            "feat_biotic_opp_negative": self.raw_opportunity("neg", sp, co),
        }


def with_embedding(x, sp, sp_emb):
    """Add the species embedding rows of ``sp`` to a dict of leaves."""
    return {**x, "sp_emb": sp_emb[sp]}


def take(x, idx):
    """Index every leaf of a dict with the same row index."""
    return {k: v[idx] for k, v in x.items()}


@torch.no_grad()
def score_pairs(model, builder, sp, co, sp_emb, neutral=True, chunk=16384):
    """Logits and climate suitability for many pairs, chunked, eval mode.

    ``neutral=True`` is the scan mode (effort-neutral biotic part); use
    the matching ``sp_emb`` (from the effort-neutral node features).
    """
    model.eval()
    logits, gates = [], []
    for i in range(0, len(sp), chunk):
        s, c = sp[i:i + chunk], co[i:i + chunk]
        lg, g = model(with_embedding(builder.build(s, c, neutral=neutral),
                                     s, sp_emb))
        logits.append(lg)
        gates.append(g)
    return torch.cat(logits), torch.cat(gates)


@torch.no_grad()
def ensemble_logits(members, builder, sp, co, raw=False, chunk=16384):
    """Mean logit over the ensemble members, chunked, eval mode.

    Each member is a dict with ``model``, ``sp_emb`` (scan mode) and
    ``sp_emb_raw`` (actual effort features). ``raw=True`` scores with the
    actual effort features (only meaningful with "neutralize").
    """
    key = "sp_emb_raw" if raw else "sp_emb"
    out = []
    for i in range(0, len(sp), chunk):
        s, c = sp[i:i + chunk], co[i:i + chunk]
        leaves = builder.build(s, c, neutral=not raw)
        total = 0.0
        for m in members:
            m["model"].eval()
            total = total + m["model"](with_embedding(leaves, s,
                                                      m[key]))[0]
        out.append(total / len(members))
    return torch.cat(out)


# ======================================================================
# 1b. Positive-unlabeled loss and self-supervised pre-training
# ======================================================================
def pu_risk(logits, is_pos, weight, prior, region, neg_ratio=1.0):
    """Non-negative positive-unlabeled risk with a class prior per region.

    The pairs sampled as "negatives" are unlabeled: a share ``prior`` of
    them are invasions that have not happened or not been recorded yet.
    Treating them all as negatives teaches the model to score such pairs
    low. The PU risk (du Plessis et al. 2015; non-negative form of Kiryo
    et al. 2017) removes the expected contribution of those hidden
    positives from the loss of the unlabeled pairs, per region::

        negatives_r = max(0, (mean_u[l-] - prior_r * mean_p[l-])
                             / (1 - prior_r))
        risk = (mean_p[l+] + neg_ratio * sum_r share_r * negatives_r)
               / (1 + neg_ratio)

    with l+ / l- the logistic loss of calling a pair positive / negative,
    mean_p over the (weighted) positives of the region, mean_u over its
    unlabeled pairs and share_r the region's share of the unlabeled
    pairs. Positives keep their full weight, so with ``prior`` = 0 this
    is the usual weighted cross-entropy without label smoothing. A
    region without positives in the batch uses the batch-wide mean_p.

    Parameters
    ----------
    logits : torch.Tensor, shape (b,)
    is_pos : torch.BoolTensor, shape (b,)
        True for labelled positives, False for unlabeled pairs.
    weight : torch.Tensor, shape (b,)
        Pair weights (used for the positives).
    prior : torch.Tensor, shape (n_regions,)
        Assumed share of hidden positives among the unlabeled pairs of
        each region, in [0, 1).
    region : torch.LongTensor, shape (b,)
        Region index of each pair's location.
    neg_ratio : float
        Weight of the negative part relative to the positive part
        (the number of unlabeled pairs per positive).
    """
    loss_pos = F.softplus(-logits)
    loss_neg = F.softplus(logits)
    unl = ~is_pos
    if not bool(is_pos.any()) or not bool(unl.any()):
        loss = torch.where(is_pos, loss_pos, loss_neg)
        return (loss * weight).sum() / weight.sum()
    w_pos = weight[is_pos]
    pos_term = (w_pos * loss_pos[is_pos]).sum() / w_pos.sum()
    zeros = logits.new_zeros(len(prior))
    w_region = zeros.index_add(0, region[is_pos], w_pos)
    pos_as_neg = zeros.index_add(0, region[is_pos],
                                 w_pos * loss_neg[is_pos])
    batch_mean = (w_pos * loss_neg[is_pos]).sum() / w_pos.sum()
    pos_as_neg = torch.where(w_region > 0,
                             pos_as_neg / w_region.clamp(min=1e-12),
                             batch_mean)
    n_unl = zeros.index_add(0, region[unl], torch.ones_like(loss_neg[unl]))
    unl_mean = (zeros.index_add(0, region[unl], loss_neg[unl])
                / n_unl.clamp(min=1))
    negatives = ((unl_mean - prior * pos_as_neg) / (1 - prior)).clamp(min=0)
    neg_term = (n_unl / n_unl.sum() * negatives).sum()
    return (pos_term + neg_ratio * neg_term) / (1 + neg_ratio)


def pretrain_on_links(encoder, x_dict, edge_index_dict, edge_weight_dict=None,
                      epochs=30, lr=1e-3, holdout=0.15, max_edges=200_000,
                      seed=0, verbose=True):
    """Self-supervised pre-training of the graph encoder on GloBI links.

    Each epoch a share ``holdout`` of the species-species interactions is
    hidden from the graph, and the encoder is trained to tell the hidden
    interactions from false ones (a relation-specific product of the two
    species embeddings). The encoder thus learns which kinds of species
    interact before it sees a single invasion record. No invasion label
    is used, so nothing leaks into the later cross-validation.

    Two details keep the task honest:

    * An interaction is hidden in **both directions and in every
      relation** at once (the graph stores reversed copies, e.g. a
      resource edge also as an enemy edge), otherwise the copy would give
      the answer away.
    * False partners are the true partners of other hidden interactions,
      shuffled, so a species cannot be recognised as a partner merely
      because it has many records (which is documentation effort).

    Parameters
    ----------
    encoder : RelationalEcologicalGNN
        Trained in place.
    x_dict, edge_index_dict : dict
        Node features and edges, on the encoder's device.
    edge_weight_dict : dict, optional
        Edge weights per edge type.
    epochs, lr : int, float
    holdout : float
        Share of interactions hidden per epoch.
    max_edges : int
        Hidden edges scored per relation and epoch (memory bound).
    seed : int
    verbose : bool

    Returns
    -------
    list of dict
        ``epoch`` and ``loss`` per epoch (empty if there is nothing to
        pre-train on).
    """
    relations = [et for et in edge_index_dict
                 if et[0] == "species" and et[2] == "species"
                 and not et[1].startswith(TAXONOMY_PREFIXES)]
    if epochs <= 0 or not relations:
        return []
    weights = edge_weight_dict or {}
    device = next(encoder.parameters()).device
    n_species = x_dict["species"].shape[0]
    dim = encoder.sp_projector.out_features
    relation_vec = nn.Parameter(torch.ones(len(relations), dim,
                                           device=device))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + [relation_vec],
                                 lr=lr)
    gen = torch.Generator().manual_seed(seed)
    prime = 2147483647
    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        # one random number per unordered species pair, shared by all
        # relations and both directions
        a, b = (int(v) for v in torch.randint(1, prime, (2,), generator=gen))
        message_edges, message_weights, hidden = {}, {}, []
        for et, ei in edge_index_dict.items():
            w = weights.get(et)
            if et not in relations:
                message_edges[et] = ei
                if w is not None:
                    message_weights[et] = w
                continue
            lo, hi = torch.minimum(ei[0], ei[1]), torch.maximum(ei[0], ei[1])
            pair = (lo * n_species + hi) % prime
            hide = ((pair * a + b) % prime).float() / prime < holdout
            message_edges[et] = ei[:, ~hide]
            if w is not None:
                message_weights[et] = w[~hide]
            held = ei[:, hide]
            if held.shape[1] > max_edges:
                pick = torch.randperm(held.shape[1], generator=gen)[:max_edges]
                held = held[:, pick.to(device)]
            hidden.append((relations.index(et), held))
        optimizer.zero_grad()
        emb = encoder(x_dict, message_edges, message_weights)["species"]
        loss, n_terms = 0.0, 0
        for rel, held in hidden:
            if held.shape[1] < 2:
                continue
            src, dst = held[0], held[1]
            fake = dst[torch.randperm(len(dst), generator=gen).to(device)]
            left = emb[src] * relation_vec[rel]
            loss = (loss + F.softplus(-(left * emb[dst]).sum(dim=-1)).mean()
                    + F.softplus((left * emb[fake]).sum(dim=-1)).mean())
            n_terms += 1
        if n_terms == 0:
            break
        loss = loss / n_terms
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss)})
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"   pre-training epoch {epoch:3d} | link loss "
                  f"{float(loss):.4f} (chance = {2 * math.log(2):.3f})")
    encoder.eval()
    return history


# ======================================================================
# 2. Scanner: ranking, uncertainty, contributions, GNN saliency
# ======================================================================
WINDOW_LABELS = {"full": "1995-2024", "recent": "2015-2024"}
AGG_LABELS = {"mean": "mean", "max": "strongest"}
SCAN_INFO_COLUMNS = ["gbif_id", "taxon_name", "group", "kingdom", "phylum",
                     "class", "order", "family",
                     "n_alien_locations", "n_native_locations",
                     "data_support", "data_support_class",
                     "n_range_locations", "n_partners", "n_pos_partners",
                     "n_neg_partners", "native_range_known",
                     "gbif_fetch_failed"]


def channel_label(name):
    """'plants_for_planting__recent' -> 'plants for planting (2015-2024)'."""
    base, _, window = name.partition("__")
    return f"{base.replace('_', ' ')} ({WINDOW_LABELS.get(window, window)})"


def leaf_feature_table(meta):
    """One row per explained input element: leaf, index, block, label, key.

    A missing ``index`` (NaN) means the leaf is summed into one
    contribution.
    """
    rows = []

    def add(leaf, items, block):
        for i, (name, key) in enumerate(items):
            rows.append((leaf, i, block, name, key))

    niche = meta["niche_cols"]
    add("target_climate", [(f"target: {pretty_feature(c)}",
                            f"attr_climate_target_{c}") for c in niche],
        "climate")
    add("niche_mean", [(f"species niche (mean): {pretty_feature(c)}",
                        f"attr_climate_niche_mean_{c}") for c in niche],
        "climate")
    add("niche_std", [(f"species niche (spread): {pretty_feature(c)}",
                       f"attr_climate_niche_std_{c}") for c in niche],
        "climate")
    add("koppen", [("Koppen-Geiger profile overlap", "attr_koppen_overlap"),
                   ("Koppen-Geiger class coverage", "attr_koppen_coverage"),
                   ("Koppen-Geiger data available",
                    "attr_koppen_available")], "climate")
    radius = int(meta.get("neighbour_radius_km", 1000))
    add("proximity", [
        ("closeness of nearest established range",
         "attr_proximity_nearest_range"),
        (f"range locations within {radius} km",
         "attr_proximity_n_range_neighbours"),
        ("share of range in the same region",
         "attr_proximity_same_region_share"),
        ("size of the known range", "attr_proximity_range_size")]
        + ([("native biota of the range resembles the target (mean)",
             "attr_proximity_biota_similarity_mean"),
            ("native biota of the range resembles the target (closest)",
             "attr_proximity_biota_similarity_max")]
           if meta.get("location_graph") else []),
        "spread")
    add("context", [(pretty_feature(c), f"attr_context_{c}")
                    for c in meta["context_cols"]], "location")
    chans = meta["trade_channel_names"]
    add("trade_import", [(f"import: {channel_label(c)}",
                          f"attr_trade_import_{c}") for c in chans], "trade")
    add("trade_export", [(f"routed from range: {channel_label(c)}",
                          f"attr_trade_export_{c}") for c in chans], "trade")
    add("trade_source", [
        (f"range exporter strength ({AGG_LABELS.get(a, a)}): "
         f"{channel_label(c)}", f"attr_trade_source_{a}_{c}")
        for a in meta["src_aggs"] for c in chans], "trade")
    add("biotic", [
        ("helpful partners present", "attr_biotic_opp_positive"),
        ("harmful partners present", "attr_biotic_opp_negative"),
        ("has known helpful partners", "attr_biotic_has_positive_partners"),
        ("has known harmful partners", "attr_biotic_has_negative_partners"),
        ("number of helpful partners", "attr_biotic_n_positive_partners"),
        ("number of harmful partners", "attr_biotic_n_negative_partners")],
        "biotic")
    rows.append(("sp_emb", None, "relational", "species graph embedding",
                 "attr_relational_ecology_species_gnn"))
    rows.append(("co_latent", None, "relational",
                 "location latent profile",
                 "attr_relational_ecology_country_latent"))
    if meta.get("location_graph"):
        rows.append(("co_graph", None, "relational",
                     "profile of connected locations (trade, distance, "
                     "native biota)",
                     "attr_relational_ecology_location_graph"))
    return pd.DataFrame(rows, columns=["leaf", "index", "block", "name",
                                       "feature_key"])


class Scanner:
    """Score, explain and map an ensemble (live or re-loaded).

    ``members``: list of dicts with ``model``, ``encoder``, ``sp_emb``
    (scan mode), ``sp_emb_raw`` (actual effort features) and
    ``baseline`` (Integrated-Gradients baseline leaves).
    """

    def __init__(self, members, builder, exclude_mask, eligible_mask,
                 meta, x_dict=None, edge_index_dict=None, n_mc_samples=30,
                 n_ig_steps=50, saliency_top_k=15, gbif_sleep_s=0.2):
        self.n_mc_samples = int(n_mc_samples)
        self.n_ig_steps = int(n_ig_steps)
        self.saliency_top_k = int(saliency_top_k)
        self.gbif_sleep_s = float(gbif_sleep_s)
        self.members = members
        for m in members:
            m["model"].eval()
            if m.get("encoder") is not None:
                m["encoder"].eval()
        self.builder = builder
        self.exclude = exclude_mask
        self.eligible = eligible_mask
        self.meta = meta
        self.x_dict = x_dict
        self.edges = edge_index_dict
        self.device = members[0]["sp_emb"].device
        self.co_to_idx = {c: i for i, c in
                          enumerate(meta["all_country_names"])}
        self.info = meta["species_info"].reset_index(drop=True)
        self.features = leaf_feature_table(meta)
        self.correct_effort = meta.get("effort_correction") == "neutralize"
        self._in_csr = None

    # ------------------------------------------------------------ scoring
    def location_index(self, location):
        """Row number of a SInAS location name."""
        if location not in self.co_to_idx:
            raise ValueError(f"Unknown location: {location!r}")
        return self.co_to_idx[location]

    def rank(self, location):
        """All candidates of a location, sorted by the ensemble score."""
        c = self.location_index(location)
        cand = (self.eligible & ~self.exclude[:, c]).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            raise ValueError(f"No candidates for {location}.")
        co = torch.full_like(cand, c)
        logits = ensemble_logits(self.members, self.builder, cand, co)
        out = pd.DataFrame({
            "sp_idx": cand.cpu().numpy(),
            "score_deterministic": torch.sigmoid(logits).cpu().numpy()})
        if self.correct_effort:
            raw = ensemble_logits(self.members, self.builder, cand, co,
                                  raw=True)
            out["score_effort_raw"] = torch.sigmoid(raw).cpu().numpy()
        else:
            out["score_effort_raw"] = out["score_deterministic"]
        return out.sort_values("score_deterministic", ascending=False,
                               ignore_index=True)

    @torch.no_grad()
    def mc_dropout(self, sp, co, n_samples=None):
        """MC dropout in every member.

        Returns (mean, total SD, SE, SD between member means) of the
        probability, as numpy arrays. ``n_samples`` defaults to the
        scanner's ``n_mc_samples``.
        """
        n_samples = n_samples or self.n_mc_samples
        leaves = self.builder.build(sp, co, neutral=True)
        per_member = []
        for m in self.members:
            model = m["model"]
            model.eval()
            dropouts = [d for d in model.modules()
                        if isinstance(d, nn.Dropout)]
            for d in dropouts:
                d.train()
            x = with_embedding(leaves, sp, m["sp_emb"])
            per_member.append(torch.stack(
                [torch.sigmoid(model(x)[0]) for _ in range(n_samples)]))
            for d in dropouts:
                d.eval()
        probs = torch.stack(per_member)  # (members, passes, pairs)
        flat = probs.reshape(-1, probs.shape[-1])
        sd = flat.std(dim=0)
        between = (probs.mean(dim=1).std(dim=0) if len(self.members) > 1
                   else torch.zeros_like(sd))
        return (flat.mean(dim=0).cpu().numpy(), sd.cpu().numpy(),
                (sd / math.sqrt(flat.shape[0])).cpu().numpy(),
                between.cpu().numpy())

    # ------------------------------------------------- feature contributions
    def explain(self, sp_i, co_i, n_steps=None):
        """Integrated Gradients averaged over the members.

        Returns (DataFrame, summary dict). The averaged contributions add
        up to mean logit(pair) - mean logit(baseline) over the members.
        ``n_steps`` defaults to the scanner's ``n_ig_steps``.
        """
        n_steps = n_steps or self.n_ig_steps
        sp = torch.tensor([sp_i], device=self.device)
        co = torch.tensor([co_i], device=self.device)
        leaves = self.builder.build(sp, co, neutral=True)
        alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps,
                                device=self.device).unsqueeze(1)
        names = [k for k in LEAF_NAMES if k in self.members[0]["baseline"]]
        attr = {k: 0.0 for k in names}
        logit_sum, base_sum = 0.0, 0.0
        for m in self.members:
            model, base = m["model"], m["baseline"]
            model.eval()
            actual = with_embedding(leaves, sp, m["sp_emb"])
            path = {k: (base[k] + alphas * (actual[k] - base[k])
                        ).requires_grad_(True) for k in names}
            logits, _ = model(path, detach_gate=False)
            grads = torch.autograd.grad(logits.sum(),
                                        [path[k] for k in names],
                                        allow_unused=True)
            for k, g in zip(names, grads):
                g = torch.zeros_like(path[k]) if g is None else g
                attr[k] = attr[k] + (g.mean(dim=0)
                                     * (actual[k][0] - base[k][0])).detach()
            with torch.no_grad():
                logit_sum += float(model(actual, detach_gate=False)[0])
                base_sum += float(model(base, detach_gate=False)[0])
        n = len(self.members)
        attr = {k: (v / n).cpu().numpy() for k, v in attr.items()}
        logit, base_logit = logit_sum / n, base_sum / n
        values = [0.0 if leaf not in attr
                  else float(attr[leaf].sum()) if pd.isna(idx)
                  else float(attr[leaf][int(idx)])
                  for leaf, idx in zip(self.features["leaf"],
                                       self.features["index"])]
        table = self.features.assign(value=values)
        summary = {"probability": 1 / (1 + math.exp(-logit)),
                   "baseline_probability": 1 / (1 + math.exp(-base_logit)),
                   "ig_completeness_gap": (logit - base_logit) - sum(values)}
        return table, summary

    # ------------------------------------------------------- GNN saliency
    def _in_neighbours(self, node_type, idx):
        """[(src_type, src_idx, relation)] of edges ending in a node."""
        if self._in_csr is None:
            self._in_csr = {}
            for et, ei in self.edges.items():
                dst = ei[1].cpu().numpy()
                order = np.argsort(dst, kind="stable")
                self._in_csr[et] = (dst[order], ei[0].cpu().numpy()[order])
        out = []
        for (st, rel, dt), (dst, src) in self._in_csr.items():
            if dt != node_type:
                continue
            lo, hi = np.searchsorted(dst, [idx, idx + 1])
            out += [(st, int(s), rel) for s in src[lo:hi]]
        return out

    def node_name(self, node_type, idx):
        """Readable name of a graph node."""
        gid = int(self.meta["graph_ids"][node_type][idx])
        if node_type == "species":
            return str(self.info.at[idx, "taxon_name"])
        name = self.meta["taxon_node_names"].get(gid)
        return f"{node_type} {name}" if name else f"{node_type} {gid}"

    def gnn_saliency(self, location, sp_list, top_k=None):
        """Node saliency maps for several focal species of one location.

        Gradient x input through each member's GNN, averaged over the
        members. Returns (nodes DataFrame, edges DataFrame). ``top_k``
        defaults to the scanner's ``saliency_top_k``.
        """
        top_k = top_k or self.saliency_top_k
        if self.x_dict is None or any(m.get("encoder") is None
                                      for m in self.members):
            raise RuntimeError("This scanner has no GNN (graph not saved).")
        c = self.location_index(location)
        leaves = {k: v.detach().clone().requires_grad_(True)
                  for k, v in self.x_dict.items()}
        types = list(leaves)
        sp = torch.as_tensor(sp_list, device=self.device)
        co = torch.full_like(sp, c)
        pair_leaves = self.builder.build(sp, co, neutral=True)
        acc = [dict() for _ in sp_list]  # per focal: (type, idx) -> sums
        for m in self.members:
            emb = m["encoder"](leaves, self.edges)["species"]
            logits, _ = m["model"](with_embedding(pair_leaves, sp, emb))
            for j in range(len(sp_list)):
                grads = torch.autograd.grad(
                    logits[j], [leaves[t] for t in types],
                    retain_graph=True, allow_unused=True)
                for t, g in zip(types, grads):
                    if g is None:
                        continue
                    l2 = g.norm(dim=1)
                    contrib = (g * leaves[t]).sum(dim=1)
                    nz = (l2 > 0).nonzero(as_tuple=True)[0]
                    for i, cv, lv in zip(nz.tolist(), contrib[nz].tolist(),
                                         l2[nz].tolist()):
                        entry = acc[j].setdefault((t, i), [0.0, 0.0])
                        entry[0] += cv
                        entry[1] += lv
            del emb, logits
        n_mem = len(self.members)
        node_rows, edge_rows = [], []
        for j, focal in enumerate(sp.tolist()):
            cands = [(t, i, v[0] / n_mem, v[1] / n_mem)
                     for (t, i), v in acc[j].items()]
            paths = self._paths(focal)
            total = sum(abs(cv) for _, _, cv, _ in cands) or 1.0
            cands.sort(key=lambda r: -abs(r[2]))
            keep = {("species", focal)}
            keep.update((t, i) for t, i, _, _ in cands[:top_k])
            for t, i, _, _ in cands[:top_k]:  # keep connectors of hop 2
                via = paths.get((t, i), {}).get("via")
                if via is not None:
                    keep.add(via)
            focal_id = int(self.meta["graph_ids"]["species"][focal])
            for rank_i, (t, i, cv, l2) in enumerate(cands, start=1):
                if (t, i) not in keep:
                    continue
                p = paths.get((t, i), {"hop": None, "path": ""})
                if (t, i) == ("species", focal):
                    p = {"hop": 0, "path": "focal species (own traits)"}
                node_rows.append({
                    "target_country": location, "gbif_id": focal_id,
                    "taxon_name": self.node_name("species", focal),
                    "node_type": t,
                    "node_gbif_id": int(self.meta["graph_ids"][t][i]),
                    "node_name": self.node_name(t, i),
                    "hop": p["hop"], "relation_path": p["path"],
                    "saliency_l2": l2, "contribution": cv,
                    "contribution_share": cv / total,
                    "saliency_rank": rank_i,
                    "in_top_k": rank_i <= top_k})
            for (t, i), p in paths.items():
                if (t, i) in keep and p.get("edge"):
                    (dt, di), rel = p["edge"]
                    if (dt, di) in keep:
                        edge_rows.append({
                            "target_country": location, "gbif_id": focal_id,
                            "src_type": t,
                            "src_gbif_id": int(self.meta["graph_ids"][t][i]),
                            "dst_type": dt,
                            "dst_gbif_id": int(self.meta["graph_ids"][dt][di]),
                            "relation": rel})
        return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)

    def _paths(self, focal):
        """Shortest incoming path (<= 2 hops) from every node to ``focal``."""
        target = ("species", focal)
        name = self.node_name
        paths = {}
        for st, si, rel in self._in_neighbours("species", focal):
            if (st, si) != target and (st, si) not in paths:
                paths[(st, si)] = {
                    "hop": 1, "via": None, "edge": (target, rel),
                    "path": f"{name(st, si)} -{rel}-> focal"}
        for (mt, mi), p1 in list(paths.items()):
            rel1 = p1["edge"][1]
            for st, si, rel2 in self._in_neighbours(mt, mi):
                if (st, si) == target or (st, si) in paths:
                    continue
                paths[(st, si)] = {
                    "hop": 2, "via": (mt, mi), "edge": ((mt, mi), rel2),
                    "path": (f"{name(st, si)} -{rel2}-> {name(mt, mi)} "
                             f"-{rel1}-> focal")}
        return paths

    # ----------------------------------------------------------- full scan
    def scan(self, location, top_n=500, explain_n=100, saliency_n=25,
             gbif_code=None, checks=None):
        """Ranked list, contributions and saliency maps for one location.

        ``checks`` (a :class:`ChecklistChecks`) flags or drops candidates
        that are native or already on the GRIIS checklist *before* the
        list is cut to ``top_n``, so dropped species do not leave gaps.
        ``gbif_code`` (ISO-2) additionally drops candidates with live
        GBIF records in that country.

        Returns a dict of DataFrames: scan, contributions, saliency_nodes,
        saliency_edges, and checks_dropped (candidates removed by the
        checks, with the reason).
        """
        c = self.location_index(location)
        ranked = self.rank(location)
        dropped = pd.DataFrame()
        if checks is not None:
            ranked, dropped = checks.apply(ranked, location, self.info,
                                           top_n)
        if gbif_code:
            ranked = verify_absent_in_gbif(ranked.head(top_n * 5), self.info,
                                           gbif_code, top_n,
                                           self.gbif_sleep_s)
        check(len(ranked) > 0, f"No candidates left for {location} after "
                               "the checks.")
        ranked = ranked.head(top_n).reset_index(drop=True)
        check_cols = [col for col in ranked.columns if col in CHECK_COLUMNS]
        sp = torch.as_tensor(ranked["sp_idx"].values, device=self.device)
        co = torch.full_like(sp, c)
        mean, sd, se, between = self.mc_dropout(sp, co)
        described = {k: v.cpu().numpy() for k, v in
                     self.builder.describe(sp, co).items()}
        scan = pd.DataFrame({
            "sp_idx": ranked["sp_idx"].values,
            "probability": mean, "uncertainty": sd, "uncertainty_se": se,
            "uncertainty_ensemble": between,
            "score_deterministic": ranked["score_deterministic"].values,
            "score_effort_raw": ranked["score_effort_raw"].values,
            **described})
        scan["effort_shift"] = (scan["score_deterministic"]
                                - scan["score_effort_raw"])
        if "gbif_occurrence_count" in ranked.columns:
            scan["gbif_occurrence_count"] = (
                ranked["gbif_occurrence_count"].values)
        for col in check_cols:
            scan[col] = ranked[col].values
        scan = scan.sort_values("probability", ascending=False,
                                ignore_index=True)
        scan.insert(0, "rank", np.arange(1, len(scan) + 1))
        info_cols = [col for col in SCAN_INFO_COLUMNS
                     if col in self.info.columns]
        scan = pd.concat([scan[["rank"]],
                          self.info.loc[scan["sp_idx"], info_cols]
                          .reset_index(drop=True),
                          scan.drop(columns="rank")], axis=1)
        scan.insert(3, "target_country", location)

        contrib_frames, extra = [], []
        for sp_i in tqdm(scan["sp_idx"].head(explain_n), desc=f"IG {location}",
                         leave=False):
            table, summary = self.explain(int(sp_i), c)
            gid = int(self.meta["graph_ids"]["species"][sp_i])
            contrib_frames.append(table.drop(columns=["leaf", "index"])
                                  .assign(gbif_id=gid, **summary))
            block_sum = table.groupby("block")["value"].sum()
            top = table.loc[table["value"].abs().nlargest(len(table)).index]
            extra.append({
                "sp_idx": sp_i,
                "top_driver": next((n for n, v in zip(top["name"],
                                                      top["value"]) if v > 0),
                                   ""),
                "top_suppressor": next((n for n, v in zip(top["name"],
                                                          top["value"])
                                        if v < 0), ""),
                "baseline_probability": summary["baseline_probability"],
                "ig_completeness_gap": summary["ig_completeness_gap"],
                **{f"block_contribution_{b}": v
                   for b, v in block_sum.items()}})
        if extra:
            scan = scan.merge(pd.DataFrame(extra), on="sp_idx", how="left")
        scan["has_contributions"] = scan.index < explain_n
        contributions = (pd.concat(contrib_frames, ignore_index=True)
                         if contrib_frames else pd.DataFrame())
        if len(contributions):
            contributions.insert(0, "target_country", location)
            contributions = contributions[
                ["target_country", "gbif_id", "block", "name", "value",
                 "feature_key", "probability", "baseline_probability",
                 "ig_completeness_gap"]]

        nodes, edges = pd.DataFrame(), pd.DataFrame()
        has_gnn = self.x_dict is not None and all(
            m.get("encoder") is not None for m in self.members)
        focal = (scan["sp_idx"].head(saliency_n).astype(int).tolist()
                 if has_gnn else [])
        if focal:
            parts = [self.gnn_saliency(location, focal[i:i + 5])
                     for i in range(0, len(focal), 5)]
            nodes = pd.concat([p[0] for p in parts], ignore_index=True)
            edges = pd.concat([p[1] for p in parts], ignore_index=True)
        scan["has_saliency_map"] = scan.index < len(focal)
        return {"scan": scan.drop(columns="sp_idx"),
                "contributions": contributions,
                "saliency_nodes": nodes, "saliency_edges": edges,
                "checks_dropped": dropped}


def verify_absent_in_gbif(ranked, info, gbif_code, top_n, sleep_s=0.2):
    """Keep the first ``top_n`` candidates with zero live GBIF records.

    One GBIF occurrence-count request per candidate (needs ``pygbif`` and
    internet). A failed lookup keeps the row with a count of -1.
    """
    from pygbif import occurrences  # optional dependency

    kept, present, errors = [], 0, 0
    for row in tqdm(ranked.itertuples(index=False), total=len(ranked),
                    desc=f"GBIF {gbif_code}", leave=False):
        if len(kept) >= top_n:
            break
        gid = int(info.at[int(row.sp_idx), "gbif_id"])
        try:
            count = occurrences.search(taxonKey=gid, country=gbif_code,
                                       limit=0).get("count", 0)
        except Exception:  # network error: keep the row, flag it
            count, errors = -1, errors + 1
        if count > 0:
            present += 1
        else:
            kept.append({**row._asdict(), "gbif_occurrence_count": count})
        time.sleep(sleep_s)
    print(f"   GBIF {gbif_code}: kept {len(kept)}, already present "
          f"{present}, errors {errors}")
    return pd.DataFrame(kept, columns=list(ranked.columns)
                        + ["gbif_occurrence_count"])


def plot_saliency_map(nodes, edges, gbif_id, ax=None):
    """Radial saliency map: focal centre, hop 1 inner ring, hop 2 outer."""
    sub = nodes[nodes["gbif_id"] == gbif_id]
    e = edges[edges["gbif_id"] == gbif_id] if len(edges) else edges
    if sub.empty:
        print("No saliency map for this species.")
        return
    import matplotlib.pyplot as plt  # only needed for this plot

    ax = ax or plt.subplots(figsize=(9, 8))[1]
    pos = {}
    for hop, radius in ((1, 1.0), (2, 2.0)):
        ring = sub[sub["hop"] == hop]
        for k, row in enumerate(ring.itertuples()):
            angle = 2 * math.pi * k / max(len(ring), 1)
            pos[(row.node_type, row.node_gbif_id)] = (radius * math.cos(angle),
                                                      radius * math.sin(angle))
    focal = sub[sub["hop"] == 0]
    for row in focal.itertuples():
        pos[(row.node_type, row.node_gbif_id)] = (0.0, 0.0)
    for row in e.itertuples():
        a = pos.get((row.src_type, row.src_gbif_id))
        b = pos.get((row.dst_type, row.dst_gbif_id))
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", alpha=0.4,
                    lw=1)
    vmax = max(sub["contribution"].abs().max(), 1e-12)
    for row in sub.itertuples():
        xy = pos.get((row.node_type, row.node_gbif_id))
        if xy is None:
            continue
        ax.scatter(*xy, s=200 + 1600 * abs(row.contribution) / vmax,
                   c=[row.contribution], cmap="RdBu_r", vmin=-vmax,
                   vmax=vmax, edgecolors="black",
                   linewidths=3 if row.hop == 0 else 1, zorder=3)
        ax.annotate(row.node_name, xy, fontsize=8, ha="center",
                    va="bottom", xytext=(0, 12), textcoords="offset points")
    ax.set_title(f"GNN saliency: {sub['taxon_name'].iloc[0]} -> "
                 f"{sub['target_country'].iloc[0]}\nred raises, blue lowers "
                 "the risk (gradient x input); size = |contribution|")
    ax.axis("off")
    ax.set_aspect("equal")
    plt.show()


# ======================================================================
# 3. Impact data (kept out of the lists)
# ======================================================================
# Documented impact (EICAT, SEICAT and context) is information for the
# people who assess a list. It is neither an input nor an output of the
# risk model, so the lists do not carry it: notebook 07 writes one file,
# IMPACT_MASTER_NAME, next to the lists, and the viewer joins it to them
# on ``gbif_id``.
IMPACT_MASTER_NAME = "species_impacts_master.csv"
# Columns that lists and checkpoints written before this change carried.
EICAT_COLUMNS = ["eicat", "eicat_source", "eicat_confidence",
                 "eicat_mechanism", "eicat_model", "eicat_justification"]


def drop_impact_columns(info):
    """Species table without the EICAT columns of older checkpoints."""
    return info.drop(columns=[c for c in EICAT_COLUMNS
                              if c in info.columns])


def attach_eicat(info, path=None):
    """Former EICAT join; now only removes old EICAT columns.

    Kept so that notebooks 05 and 06 written before notebook 07 took
    over the impact data keep running. ``path`` is ignored. Remove the
    call when you next edit those notebooks.

    Parameters
    ----------
    info : pandas.DataFrame
        One row per species (row order and index kept).
    path : str or Path, optional
        Ignored.
    """
    warnings.warn("Impact data is no longer written into the lists; the "
                  f"viewer reads {IMPACT_MASTER_NAME} from notebook 07. "
                  "attach_eicat() can be removed.", FutureWarning,
                  stacklevel=2)
    return drop_impact_columns(info)


# ======================================================================
# 4. Checklist checks: native range and GRIIS
# ======================================================================
CHECK_COLUMNS = ["check_flags", "check_native_target",
                 "check_native_same_country", "check_native_extra",
                 "griis_listed", "griis_invasive", "griis_scope"]
CHECK_ACTIONS = ("drop", "flag", "off")
# Column names tried (case-insensitive, in this order) in the GRIIS file.
GRIIS_COLUMN_CANDIDATES = {
    "name": ["acceptedNameUsage", "species", "scientificName",
             "canonicalName", "taxon"],
    "iso3": ["countryCode_alpha3", "countryCodeAlpha3", "ISO3", "alpha3"],
    "iso2": ["countryCode_alpha2", "countryCodeAlpha2", "countryCode",
             "ISO2", "alpha2"],
    "country": ["country", "countryName", "location"],
    "invasive": ["isInvasive", "is_invasive", "invasive"],
    "kingdom": ["kingdom"],
}
_TRUE_WORDS = {"invasive", "yes", "true", "1", "y", "t"}


def canonical_binomial(name):
    """'Acer negundo L.' -> 'acer negundo'; None if not a species name.

    Authorship, infraspecific parts and hybrid signs are dropped, so
    subspecies and varieties fall on their species (as in notebook 03).
    """
    if not isinstance(name, str):
        return None
    words = name.replace("×", " ").split()
    if len(words) < 2:
        return None
    genus, epithet = words[0], words[1]
    if not (genus[:1].isupper() and genus.isalpha()):
        return None
    if not (epithet.replace("-", "").isalpha() and epithet.islower()
            and len(epithet) > 1):
        return None
    return f"{genus.lower()} {epithet}"


def build_name_index(info, audit_csv=None):
    """{canonical binomial: sorted species rows} for name-based matching.

    Names come from the accepted Backbone names in ``info`` and, if the
    name-matching audit of notebook 03 is given, from every SInAS / GloBI
    name that was matched to one of these species (synonyms).
    """
    index = defaultdict(set)
    for i, name in enumerate(info["taxon_name"]):
        key = canonical_binomial(name)
        if key:
            index[key].add(i)
    if audit_csv is not None and Path(audit_csv).exists():
        row_of = {int(g): i for i, g in enumerate(info["gbif_id"])}
        cols = pd.read_csv(audit_csv, nrows=0).columns
        use = [c for c in ("name", "canonicalName", "species_key")
               if c in cols]
        if "species_key" in use:
            audit = pd.read_csv(audit_csv, usecols=use, low_memory=False)
            audit = audit.dropna(subset=["species_key"])
            rows = audit["species_key"].astype(np.int64).map(row_of)
            for col in use[:-1]:
                for name, i in zip(audit[col], rows):
                    key = canonical_binomial(name)
                    if key and pd.notna(i):
                        index[key].add(int(i))
    return {k: sorted(v) for k, v in index.items()}


def _find_column(columns, candidates):
    """First of ``candidates`` present in ``columns`` (case-insensitive)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _read_text_table(path):
    """Read a delimited file with an unknown delimiter and encoding."""
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", dtype=str,
                               encoding=enc)
        except UnicodeDecodeError as err:
            last = err
    raise ValueError(f"Could not decode {path}: {last}")


def load_griis(path, info, audit_csv=None, columns=None):
    """Read the GRIIS Country Compendium and match it to the species list.

    GRIIS lists, per country, the species recorded as introduced there and
    flags those with evidence of impact (``isInvasive``). Species are
    matched to the scanner's species by canonical name (accepted names
    plus the synonyms of notebook 03); the kingdom settles names that
    exist in two kingdoms.

    Parameters
    ----------
    path : str or Path
        The compendium CSV (see ``06_static_scanner.ipynb``, Step 0).
    info : pandas.DataFrame
        Species table of the scanner (``gbif_id``, ``taxon_name``,
        ``kingdom``).
    audit_csv : str or Path, optional
        ``name_matching_audit.csv`` of notebook 03, for synonyms.
    columns : dict, optional
        Column names to use instead of the detected ones, with any of the
        keys of GRIIS_COLUMN_CANDIDATES, e.g. ``{"iso3": "ISO3"}``.

    Returns
    -------
    pandas.DataFrame
        ``sp_idx``, ``gbif_id``, ``iso3``, ``griis_invasive``,
        ``griis_name``; one row per species x country. ``.attrs["report"]``
        holds the match statistics.
    """
    raw = _read_text_table(path)
    found = {key: (columns or {}).get(key)
             or _find_column(raw.columns, cands)
             for key, cands in GRIIS_COLUMN_CANDIDATES.items()}
    name_cols = [c for c in ([(columns or {}).get("name")]
                             + [_find_column(raw.columns, [cand]) for cand
                                in GRIIS_COLUMN_CANDIDATES["name"]])
                 if c]
    name_cols = list(dict.fromkeys(name_cols))
    check(name_cols, f"{Path(path).name}: no species-name column found "
                     f"among {list(raw.columns)}; set GRIIS_COLUMNS.")
    check(found["iso3"] or found["iso2"] or found["country"],
          f"{Path(path).name}: no country column found among "
          f"{list(raw.columns)}; set GRIIS_COLUMNS.")
    if found["iso3"]:
        iso3 = raw[found["iso3"]].str.strip().str.upper()
    else:
        import country_converter as coco  # only needed for this case

        source = raw[found["iso2"] or found["country"]].str.strip()
        codes = {v: coco.convert(names=v, to="ISO3", not_found=None)
                 for v in source.dropna().unique()}
        iso3 = source.map(codes)
    iso3 = iso3.where(iso3.astype(str).str.len() == 3)
    invasive = (raw[found["invasive"]].fillna("").astype(str).str.strip()
                .str.lower().isin(_TRUE_WORDS) if found["invasive"]
                else pd.Series(False, index=raw.index))

    index = build_name_index(info, audit_csv)
    kingdoms = info["kingdom"].fillna("").astype(str).str.lower().to_numpy()
    kingdom = (raw[found["kingdom"]].fillna("").astype(str).str.strip()
               .str.lower() if found["kingdom"] else None)
    rows, names_out = [], []
    for pos in range(len(raw)):
        hit = None
        for col in name_cols:
            hit = index.get(canonical_binomial(raw[col].iat[pos]))
            if hit:
                names_out.append(raw[col].iat[pos])
                break
        if not hit:
            rows.append(None)
            continue
        if len(hit) > 1 and kingdom is not None:
            same = [i for i in hit if kingdoms[i] == kingdom.iat[pos]]
            hit = same or hit
        rows.append(hit)
    matched = pd.Series([r is not None for r in rows], index=raw.index)
    ok = matched & iso3.notna()
    out = pd.DataFrame({
        "sp_idx": [r for r, keep in zip(rows, ok) if keep],
        "iso3": iso3[ok].to_numpy(),
        "griis_invasive": invasive[ok].to_numpy(),
        "griis_name": [n for n, keep in zip(names_out, ok[matched])
                       if keep],
    }).explode("sp_idx", ignore_index=True)
    out["sp_idx"] = out["sp_idx"].astype(np.int64)
    out = (out.sort_values("griis_invasive", ascending=False)
           .drop_duplicates(["sp_idx", "iso3"]).reset_index(drop=True))
    out.insert(1, "gbif_id", info["gbif_id"].to_numpy()[out["sp_idx"]])
    out.attrs["report"] = {
        "file": Path(path).name, "rows": int(len(raw)),
        "rows_with_country_code": int(iso3.notna().sum()),
        "rows_matched_to_a_species": int(ok.sum()),
        "share_matched": float(ok.mean()) if len(raw) else 0.0,
        "countries": int(out["iso3"].nunique()),
        "species": int(out["sp_idx"].nunique()),
        "columns_used": {**found, "name": name_cols},
    }
    return out


class ChecklistChecks:
    """Native-range and GRIIS checks for the candidates of a location.

    A horizon list should hold species that are *not yet* in the
    location. The model already leaves out every species with a SInAS
    record (any status) or a GBIF presence there; these checks catch what
    that misses:

    =========================== =========================================
    Flag                        Meaning
    =========================== =========================================
    ``native_target``           SInAS lists the species as native in the
                                location itself (should not happen: it
                                means the checkpoint and the SInAS file
                                disagree)
    ``native_same_country``     native in another SInAS location with the
                                same ISO3 code (e.g. the mainland when an
                                island group is scanned)
    ``native_extra``            native in the location according to the
                                optional extra checklist
    ``griis_listed``            on the GRIIS checklist of the location's
                                country: already introduced there
    ``griis_invasive``          GRIIS records evidence of impact there
    =========================== =========================================

    Each group has an action: ``"drop"`` (remove the candidate),
    ``"flag"`` (keep it, with the flag in ``check_flags``) or ``"off"``.
    GRIIS works per country. For a location that shares its ISO3 code
    with other SInAS locations (a sub-national unit or its mainland) a
    GRIIS record cannot be assigned to one of them, so it is only ever
    flagged there (``griis_scope = "country"``), never dropped.

    Parameters
    ----------
    info : pandas.DataFrame
        Species table of the scanner (row = species index).
    location_names : list of str
        SInAS location names in model order.
    location_iso3 : list of str or None, optional
        ISO3 code per location (None where unknown).
    native_pairs : tuple of arrays, optional
        ``(species rows, location rows)`` of the SInAS native records.
    griis : pandas.DataFrame, optional
        Output of :func:`load_griis`.
    extra_native : pandas.DataFrame, optional
        Extra native checklist: ``gbif_id`` plus ``location`` (SInAS
        name) or ``iso3``.
    native_action, same_country_action, griis_action : str
        One of CHECK_ACTIONS.
    """

    def __init__(self, info, location_names, location_iso3=None,
                 native_pairs=None, griis=None, extra_native=None,
                 native_action="drop", same_country_action="flag",
                 griis_action="drop"):
        for action in (native_action, same_country_action, griis_action):
            check(action in CHECK_ACTIONS,
                  f"Check actions must be one of {CHECK_ACTIONS}.")
        self.actions = {"native": native_action,
                        "same_country": same_country_action,
                        "griis": griis_action}
        self.names = list(location_names)
        self.n_loc = len(self.names)
        self.loc_index = {name: i for i, name in enumerate(self.names)}
        iso = list(location_iso3) if location_iso3 is not None else []
        iso += [None] * (self.n_loc - len(iso))
        self.iso3 = [(str(v).strip().upper() if isinstance(v, str)
                      and len(v.strip()) == 3 else None) for v in iso]
        by_iso = defaultdict(list)
        for i, code in enumerate(self.iso3):
            if code:
                by_iso[code].append(i)
        self.siblings = {i: [j for j in by_iso.get(code, []) if j != i]
                         for i, code in enumerate(self.iso3)}
        self.native_codes = self._codes(native_pairs)
        self.extra_codes = self._codes(self._extra_pairs(info, extra_native,
                                                         by_iso))
        self.griis_listed, self.griis_invasive = {}, {}
        if griis is not None and len(griis):
            for code, grp in griis.groupby("iso3"):
                self.griis_listed[code] = np.unique(grp["sp_idx"])
                self.griis_invasive[code] = np.unique(
                    grp.loc[grp["griis_invasive"], "sp_idx"])
        self.has_griis = bool(self.griis_listed)
        self.report = {
            "actions": dict(self.actions),
            "native_records": int(len(self.native_codes)),
            "extra_native_records": int(len(self.extra_codes)),
            "griis": (dict(griis.attrs.get("report", {}))
                      if griis is not None else None),
            "locations_without_iso3": [n for n, c in zip(self.names,
                                                         self.iso3)
                                       if c is None],
        }
        self.log = []

    def _codes(self, pairs):
        """Sorted array of ``species * n_locations + location`` codes."""
        if pairs is None:
            return np.empty(0, dtype=np.int64)
        sp = np.asarray(pairs[0], dtype=np.int64)
        co = np.asarray(pairs[1], dtype=np.int64)
        return np.unique(sp * self.n_loc + co)

    def _extra_pairs(self, info, extra, by_iso):
        """(species rows, location rows) of the extra native checklist."""
        if extra is None or not len(extra):
            return None
        check("gbif_id" in extra.columns
              and ({"location", "iso3"} & set(extra.columns)),
              "The extra native checklist needs gbif_id and location or "
              "iso3.")
        row_of = {int(g): i for i, g in enumerate(info["gbif_id"])}
        sp_rows, co_rows = [], []
        for rec in extra.to_dict("records"):
            sp = row_of.get(int(rec["gbif_id"])) if pd.notna(
                rec["gbif_id"]) else None
            if sp is None:
                continue
            loc = rec.get("location")
            targets = ([self.loc_index[loc]] if loc in self.loc_index
                       else by_iso.get(str(rec.get("iso3", "")).upper(),
                                       []))
            sp_rows += [sp] * len(targets)
            co_rows += targets
        return np.asarray(sp_rows), np.asarray(co_rows)

    def _has(self, codes, sp, locations):
        """True where a species has a record in any of ``locations``."""
        out = np.zeros(len(sp), dtype=bool)
        if len(codes):
            for loc in locations:
                out |= np.isin(sp * self.n_loc + loc, codes)
        return out

    def flags(self, sp, location):
        """Check columns for species rows ``sp`` in one location.

        Returns
        -------
        pandas.DataFrame
            CHECK_COLUMNS, one row per entry of ``sp``.
        """
        c = self.loc_index[location]
        sp = np.asarray(sp, dtype=np.int64)
        code = self.iso3[c]
        siblings = self.siblings.get(c, [])
        empty = np.empty(0, dtype=np.int64)
        listed = np.isin(sp, self.griis_listed.get(code, empty))
        out = pd.DataFrame({
            "check_native_target": self._has(self.native_codes, sp, [c]),
            "check_native_same_country": self._has(self.native_codes, sp,
                                                   siblings),
            "check_native_extra": self._has(self.extra_codes, sp, [c]),
            "griis_listed": listed,
            "griis_invasive": np.isin(sp, self.griis_invasive.get(code,
                                                                  empty)),
        })
        out["griis_scope"] = ("" if not (self.has_griis and code)
                              else "country" if siblings else "location")
        labels = pd.Series("", index=out.index)
        for name, col, on in (
                ("native_target", "check_native_target",
                 self.actions["native"]),
                ("native_extra", "check_native_extra",
                 self.actions["native"]),
                ("native_same_country", "check_native_same_country",
                 self.actions["same_country"]),
                ("griis_listed", "griis_listed", self.actions["griis"]),
                ("griis_invasive", "griis_invasive",
                 self.actions["griis"])):
            if on != "off":
                labels = labels.where(~out[col], labels + name + ";")
        out["check_flags"] = labels.str.rstrip(";")
        return out[CHECK_COLUMNS]

    def apply(self, ranked, location, info, top_n):
        """Flag and drop candidates of a ranked list.

        Parameters
        ----------
        ranked : pandas.DataFrame
            Output of ``Scanner.rank`` (sorted, with ``sp_idx``).
        location : str
        info : pandas.DataFrame
            Species table (for the names in the dropped list).
        top_n : int
            List length; the summary counts what the checks removed from
            the first ``top_n`` places.

        Returns
        -------
        (pandas.DataFrame, pandas.DataFrame)
            The kept rows with the check columns, and the dropped
            candidates that would have been in the top ``top_n``.
        """
        flags = self.flags(ranked["sp_idx"].to_numpy(), location)
        flags.index = ranked.index
        act = self.actions
        scope = flags["griis_scope"].iat[0] if len(flags) else ""
        reasons = pd.Series("", index=ranked.index)
        if act["native"] == "drop":
            reasons = reasons.where(~flags["check_native_target"],
                                    "native in the location (SInAS)")
            reasons = reasons.where(
                ~(flags["check_native_extra"] & (reasons == "")),
                "native in the location (extra checklist)")
        if act["same_country"] == "drop":
            reasons = reasons.where(
                ~(flags["check_native_same_country"] & (reasons == "")),
                "native elsewhere in the same country (SInAS)")
        if act["griis"] == "drop" and scope == "location":
            reasons = reasons.where(
                ~(flags["griis_listed"] & (reasons == "")),
                "on the GRIIS checklist of the location")
        drop = reasons != ""
        position = np.arange(1, len(ranked) + 1)
        kept_pos = position[~drop.to_numpy()]
        cutoff = (kept_pos[min(top_n, len(kept_pos)) - 1]
                  if len(kept_pos) else len(ranked))
        in_top = drop.to_numpy() & (position <= cutoff)
        sp_top = ranked["sp_idx"].to_numpy()[in_top]
        dropped = pd.DataFrame({
            "target_country": location,
            "gbif_id": info["gbif_id"].to_numpy()[sp_top],
            "taxon_name": info["taxon_name"].to_numpy()[sp_top],
            "rank_before_checks": position[in_top],
            "score_deterministic":
                ranked["score_deterministic"].to_numpy()[in_top],
            "reason": reasons.to_numpy()[in_top],
            "griis_invasive": flags["griis_invasive"].to_numpy()[in_top],
        })
        kept = pd.concat([ranked, flags], axis=1)[~drop]
        head = kept.head(top_n)
        self.log.append({
            "target_country": location,
            "iso3": self.iso3[self.loc_index[location]],
            "griis_scope": scope,
            "candidates": int(len(ranked)),
            "dropped_all": int(drop.sum()),
            "dropped_from_top": int(in_top.sum()),
            "dropped_native": int(dropped["reason"].str.startswith(
                "native").sum()),
            "dropped_griis": int(dropped["reason"].str.contains(
                "GRIIS").sum()),
            "flagged_in_list": int((head["check_flags"] != "").sum()),
            "flagged_native_same_country": int(
                head["check_native_same_country"].sum()),
            "flagged_griis": int(head["griis_listed"].sum()),
        })
        return kept, dropped

    def summary(self):
        """One row per scanned location: what the checks did."""
        return pd.DataFrame(self.log)


# ======================================================================
# 5. Static checkpoint loader and export writer
# ======================================================================
EXPORT_NAMES = {"scan": "scan", "contributions": "contributions",
                "saliency_nodes": "gnn_saliency_nodes",
                "saliency_edges": "gnn_saliency_edges",
                "checks_dropped": "checks_dropped"}


def load_static_scanner(path, device=None, feature_chunk=4096,
                        n_mc_samples=None, n_ig_steps=None,
                        saliency_top_k=None):
    """Rebuild a :class:`Scanner` from ``horizon_scan_static.pt``.

    The checkpoint holds the ensemble weights, the species embeddings and
    every static input, so no data file and no training is needed.

    .. warning:: The file is a Python pickle. Only load checkpoints you
       wrote yourself with notebook 05.

    Parameters
    ----------
    path : str or Path
    device : torch.device, optional
        Default: the GPU if there is one, else the CPU.
    feature_chunk : int
        Pairs per feature-building chunk (lower it to save memory).
    n_mc_samples, n_ig_steps, saliency_top_k : int, optional
        Default: the values the checkpoint was exported with.

    Returns
    -------
    (Scanner, dict)
        The scanner, and the other checkpoint contents: ``format_version``,
        ``meta``, ``config``, ``metrics``, ``model_card``, ``native_pairs``
        and ``location_iso3`` (the last three are None for format 4).
        Formats 4 and 5 (GraphSAGE encoder, no location graph) load with
        the settings they were trained with.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    version = ckpt.get("format_version")
    check(version in SUPPORTED_CHECKPOINT_FORMATS,
          f"Checkpoint format {version} is not supported (supported: "
          f"{SUPPORTED_CHECKPOINT_FORMATS}); retrain with notebook 05.")
    meta = ckpt["meta"]
    config = meta.get("config", {})
    static = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in ckpt["static"].items()}
    builder = PairFeatureBuilder(
        static, ckpt["range_mat"].float().to(device), meta["src_aggs"],
        feature_chunk, bio_neutral=ckpt["bio_neutral"].to(device))
    # n_records edge weights (format 6; older checkpoints have none)
    edge_weight = {k: v.to(device)
                   for k, v in (ckpt.get("edge_weight_dict") or {}).items()}
    members = []
    for mem in ckpt["members"]:
        model = InvasionRiskModel(ckpt["model_dims"])
        model.load_state_dict(mem["model_state"])
        encoder = RelationalEcologicalGNN(
            meta["emb_dim"], meta["edge_types"], meta["node_types"],
            meta["n_sp_features"], **meta.get("gnn", {}))
        encoder.load_state_dict(mem["encoder_state"])
        encoder.edge_weight = edge_weight
        members.append({
            "model": model.to(device), "encoder": encoder.to(device),
            "sp_emb": mem["sp_emb"].to(device),
            "sp_emb_raw": mem["sp_emb_raw"].to(device),
            "baseline": {k: v.to(device)
                         for k, v in mem["ig_baseline"].items()}})
    scanner = Scanner(
        members, builder, ckpt["exclude_mask"].to(device),
        ckpt["eligible_mask"].to(device), meta,
        x_dict={k: v.to(device) for k, v in ckpt["x_dict"].items()},
        edge_index_dict={k: v.to(device)
                         for k, v in ckpt["edge_index_dict"].items()},
        n_mc_samples=n_mc_samples or config.get("N_MC_SAMPLES", 30),
        n_ig_steps=n_ig_steps or config.get("N_IG_STEPS", 50),
        saliency_top_k=saliency_top_k or config.get("SALIENCY_TOP_K", 15),
        gbif_sleep_s=config.get("GBIF_SLEEP_S", 0.2))
    native = ckpt.get("native_pairs")
    extras = {
        "format_version": version, "meta": meta, "config": config,
        "metrics": ckpt.get("metrics", {}),
        "model_card": ckpt.get("model_card"),
        "native_pairs": (None if native is None
                         else (np.asarray(native[0]), np.asarray(native[1]))),
        "location_iso3": ckpt.get("location_iso3"),
    }
    return scanner, extras


def export_scans(scanner, locations, export_dir, top_n=500, explain_n=100,
                 saliency_n=25, checks=None, gbif_codes=None,
                 write_combined=True):
    """Write the viewer files of every location to ``export_dir``.

    Per location: ``scan_<loc>.csv``, ``contributions_<loc>.csv``,
    ``gnn_saliency_nodes_<loc>.csv``, ``gnn_saliency_edges_<loc>.csv``
    and, if the checks removed candidates, ``checks_dropped_<loc>.csv``.
    A table without rows (e.g. no saliency maps asked for) is not written.
    Once: ``feature_dictionary.csv``, ``checks_summary.csv`` (with
    checks) and, for several locations, ``*_all_locations.csv``.

    Parameters
    ----------
    scanner : Scanner
    locations : list of str
        SInAS location names.
    export_dir : str or Path
    top_n, explain_n, saliency_n : int
        List length; how many of the list get contributions; how many
        get a saliency map.
    checks : ChecklistChecks, optional
    gbif_codes : dict, optional
        ``{location: ISO-2 code}`` for the live GBIF check.
    write_combined : bool
        Also write the ``*_all_locations.csv`` files (large for many
        locations).

    Returns
    -------
    dict
        ``last``: the DataFrames of the last location; ``summary``: the
        checks summary (empty without checks).
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    gbif_codes = gbif_codes or {}
    combined = defaultdict(list)
    out = {}
    for loc in tqdm(list(locations), desc="Locations"):
        out = scanner.scan(loc, top_n=top_n, explain_n=explain_n,
                           saliency_n=saliency_n,
                           gbif_code=gbif_codes.get(loc), checks=checks)
        for kind, frame in out.items():
            name = EXPORT_NAMES[kind]
            target = export_dir / f"{name}_{slugify(loc)}.csv"
            if frame.empty:  # nothing to write; leave no stale file
                target.unlink(missing_ok=True)
                continue
            frame.to_csv(target, index=False)
            if write_combined:
                combined[name].append(frame)
    if write_combined and len(locations) > 1:
        for name, frames in combined.items():
            pd.concat(frames, ignore_index=True).to_csv(
                export_dir / f"{name}_all_locations.csv", index=False)
    scanner.features.drop(columns="index").to_csv(
        export_dir / "feature_dictionary.csv", index=False)
    summary = checks.summary() if checks is not None else pd.DataFrame()
    if len(summary):
        summary.to_csv(export_dir / "checks_summary.csv", index=False)
    return {"last": out, "summary": summary}


def write_model_card(card, export_dir, **extra):
    """Write ``model_card.json`` (the viewer's "About the model" tab).

    ``extra`` entries (e.g. ``scan`` and ``checks``) are added to a copy
    of ``card``.
    """
    card = {**(card or {}), **extra,
            "exported_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds")}
    path = Path(export_dir) / "model_card.json"
    with open(path, "w") as fh:
        json.dump(card, fh, indent=2, default=str)
    return path
