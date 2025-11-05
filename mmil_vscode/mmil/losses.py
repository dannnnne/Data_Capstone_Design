from __future__ import annotations
import torch
import torch.nn.functional as F

# =========================
# (A) BCE 기반 MIL 손실 (logit → pool → BCE)
#  - 빈 bag/마스크를 안전하게 처리
#  - 클래스 불균형용 pos_weight 지원
# =========================

def _ensure_bool(mask: torch.Tensor) -> torch.Tensor:
    return mask if mask.dtype == torch.bool else mask.bool()

def _masked_mean_logit(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _ensure_bool(mask)
    w = mask.float()
    # 마스킹된 위치는 0으로, 분모는 최소 1
    return logits.masked_fill(~mask, 0.0).sum(1) / w.sum(1).clamp_min(1.0)

def _masked_max_logit(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _ensure_bool(mask)
    x = logits.masked_fill(~mask, float("-inf"))
    out = x.max(1).values
    # 빈 bag이면 0 로짓(= 0.5 확률)로 대체
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

def _masked_topk_mean_logit(logits: torch.Tensor, mask: torch.Tensor, ratio: float) -> torch.Tensor:
    mask = _ensure_bool(mask)
    B, T = logits.shape
    x = logits.masked_fill(~mask, float("-inf"))
    k = (mask.sum(1).float() * ratio).floor().clamp_min(1).long()
    outs = []
    for b in range(B):
        if not mask[b].any():
            outs.append(torch.tensor(0., device=logits.device, dtype=logits.dtype))
        else:
            vals = torch.topk(x[b], k=int(min(k[b].item(), T))).values
            vals = vals[torch.isfinite(vals)]
            outs.append(vals.mean() if vals.numel() else torch.tensor(0., device=logits.device, dtype=logits.dtype))
    return torch.stack(outs, 0)

def _masked_lse_logit(logits: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    mask = _ensure_bool(mask)
    x = logits.masked_fill(~mask, float("-inf"))
    out = torch.logsumexp(tau * x, dim=1) / tau
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

def mil_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pool: str = "lse",
    tau: float = 5.0,
    topk_ratio: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: (B,T), labels: (B,), mask: (B,T) bool
    pool ∈ {"lse","mean","max","topk"}
    """
    if pool == "mean":
        bag_logit = _masked_mean_logit(logits, mask)
    elif pool == "max":
        bag_logit = _masked_max_logit(logits, mask)
    elif pool == "topk":
        bag_logit = _masked_topk_mean_logit(logits, mask, ratio=topk_ratio)
    else:
        bag_logit = _masked_lse_logit(logits, mask, tau=tau)

    return F.binary_cross_entropy_with_logits(bag_logit, labels.float(), pos_weight=pos_weight)


# =========================
# (B) 확률 기반 5항 컴포짓 손실
#  - mean-MIL(hinge-margin)
#  - sparsity(양성), normal suppression(음성)
#  - temporal smooth(TV/L2), modality agreement
#  - 모든 항에 mask 적용
# =========================

def _mask_float(mask: torch.Tensor) -> torch.Tensor:
    return mask.float() if mask.dtype == torch.bool else mask

def mean_over_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = _mask_float(mask)
    return (x * w).sum(1) / w.sum(1).clamp_min(1e-6)

def sum_over_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = _mask_float(mask)
    return (x * w).sum(1)

def loss_mean_mil(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    y = labels.bool()
    pos_mean = mean_over_mask(scores[y], mask[y]).mean() if y.any() else scores.sum() * 0.
    neg_mean = mean_over_mask(scores[~y], mask[~y]).mean() if (~y).any() else scores.sum() * 0.
    d = pos_mean - neg_mean
    # 마진 미만이면 벌점, 넘으면 0 → 음수 손실 방지
    return F.relu(margin - d)

def loss_sparsity_pos(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    if lam <= 0: return scores.sum() * 0.
    y = labels.bool()
    if not y.any(): return scores.sum() * 0.
    area = sum_over_mask(scores[y], mask[y]).mean()
    return lam * area

def loss_normal_supp(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    if lam <= 0: return scores.sum() * 0.
    y = labels.bool()
    if (~y).any():
        m = mean_over_mask(scores[~y], mask[~y]).mean()
        return lam * m
    return scores.sum() * 0.

def loss_temporal_smooth(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, lam: float, kind: str = "tv") -> torch.Tensor:
    if lam <= 0: return scores.sum() * 0.
    y = labels.bool()
    if not y.any(): return scores.sum() * 0.
    sp, mp = scores[y], mask[y]
    diff = (sp[:, 1:] - sp[:, :-1]) * (mp[:, 1:] & mp[:, :-1]).float()
    L = diff.abs().sum(1).mean() if kind == "tv" else diff.pow(2).sum(1).mean()
    return lam * L

def loss_modality_agreement(scores_f: torch.Tensor, scores_a: torch.Tensor, scores_v: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    if lam <= 0: return scores_f.sum() * 0.
    target = 0.5 * (scores_a + scores_v)
    w = _mask_float(mask)
    num = ((scores_f - target).pow(2) * w).sum()
    den = w.sum().clamp_min(1e-6)
    return lam * (num / den)

def total_loss_from_batch(
    scores_f: torch.Tensor, scores_a: torch.Tensor, scores_v: torch.Tensor,
    labels: torch.Tensor, mask: torch.Tensor,
    lam_sp: float, lam_norm: float, lam_tv: float, lam_ag: float, margin: float = 0.0
) -> torch.Tensor:
    L_mil  = loss_mean_mil(scores_f, labels, mask, margin=margin)
    L_sp   = loss_sparsity_pos(scores_f, labels, mask, lam_sp)
    L_norm = loss_normal_supp(scores_f, labels, mask, lam_norm)
    L_tv   = loss_temporal_smooth(scores_f, labels, mask, lam_tv, kind="tv")
    L_ag   = loss_modality_agreement(scores_f, scores_a, scores_v, mask, lam_ag)
    return L_mil + L_sp + L_norm + L_tv + L_ag


# =========================
# (C) cfg로 손실 선택하는 래퍼
#  - loss_type: 'bce_mil' | 'composite'
# =========================

def compute_loss(
    cfg,
    logits_f: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    logits_a: torch.Tensor | None = None,
    logits_v: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_type = getattr(cfg, "loss_type", "bce_mil")
    if loss_type == "composite":
        # 컴포짓 손실은 확률 영역에서 정의 → 시그모이드
        scores_f = torch.sigmoid(logits_f)
        scores_a = torch.sigmoid(logits_a) if logits_a is not None else scores_f
        scores_v = torch.sigmoid(logits_v) if logits_v is not None else scores_f
        return total_loss_from_batch(
            scores_f, scores_a, scores_v, labels, mask,
            getattr(cfg, "lam_sp", 0.0),
            getattr(cfg, "lam_norm", 0.0),
            getattr(cfg, "lam_tv", 0.0),
            getattr(cfg, "lam_ag", 0.0),
            margin=getattr(cfg, "margin", 0.0),
        )
    else:
        # BCE-MIL (logit 영역)
        pos_weight = None
        if getattr(cfg, "pos_weight", None) is not None:
            pos_weight = torch.tensor([cfg.pos_weight], device=logits_f.device, dtype=logits_f.dtype)
        return mil_bce_loss(
            logits_f, labels, mask,
            pool=cfg.mil_pool, tau=cfg.mil_tau, topk_ratio=cfg.mil_topk_ratio, pos_weight=pos_weight
        )
