"""What the VQ-VAE's tokens stand for (configs/analysis/vq_tokens.yaml, jobs/vqvae/vq_tokens.sh).

On frames of the validation layouts, the token the encoder gives each tile next to what the tile shows (stable-marl's
`image` column: its kind, and the partner by colour and by its arrow as drawn, i.e. its direction relative to the
viewer's), and:
  kinds     each code's share of each kind of tile, the codes' purity and how much of the kind they tell
  agents    for the partner (colour x direction), the goal and the ego: the codes that carry them
  gallery   per kind, its most used codes, each with tiles it was given (a tile of context around them)
  codes     what each code draws on its own (the decoder on a 7x7 grid of that code)
  swaps     example frames with one tile's token replaced by the main code of each label, decoded
  maps      example frames, their reconstruction, tokens and tiles
  codebook  the codebook's 2-D PCA, by each code's main kind
Everything the figures need is saved (<out_dir>/tokens.npz, summary.json), so they redraw without the model or the data:
    python vq_tokens.py checkpoint=...                                  # compute, save, draw, log to wandb
    python vq_tokens.py stage=plot out_dir=... wandb.enabled=false      # redraw from the saved results
"""
import json
import os
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

TILE = 32                                                    # pixels per tile: 7 x 7 tiles of 224 x 224 frames
DIRECTIONS = ('up', 'right', 'down', 'left')                 # the partner's arrow as drawn: (its direction - the viewer's) mod 4
COLOURS = ('red', 'green', 'blue')
GALLERY_KINDS = ('empty', 'wall', 'goal', 'partner', 'ego')       # the kinds whose tiles the galleries keep, per code                           # an agent's colour: its pixels' strongest channel
KIND_COLOURS = {'unseen': '#ffffff', 'empty': '#222222', 'wall': '#9a9a9a', 'floor': '#6b4f2a', 'door': '#c9a227',
                'key': '#e6c200', 'ball': '#1f77b4', 'box': '#8c564b', 'goal': '#2ca02c', 'lava': '#ff7f0e',
                'partner': '#9467bd', 'ego': '#d62728'}


def label_name(value: int, kinds) -> str:
    "A tile's label (`fine` in tokens.npz: kind * 100 + 10 * colour + direction) as text."
    kind, rest = divmod(int(value), 100)
    name = kinds[kind]
    if name in ('partner', 'ego'):
        colour, direction = divmod(rest, 10)
        name += f' {COLOURS[colour]}' + (f' {DIRECTIONS[direction]}' if name == 'partner' else '')
    return name


def to_uint8(x):
    "Frames normalized to [-1, 1] (B, 3, H, W) as uint8 images (B, H, W, 3)."
    return ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


# --------------------------------------------------------------------------------------------------- computing

def frames_dataset(ds, picks):
    "The dataset's frames `picks`: (position, frame, kinds of tiles, labels), the labels as `label_name` reads them."
    import torch
    from c3jepa_wm.data.datasets import TILE_NAMES
    partner, ego = TILE_NAMES.index('partner'), TILE_NAMES.index('ego')

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return len(picks)

        def __getitem__(self, j):
            i = int(picks[j])
            frame, kinds = ds[i]                                 # opens the file in this worker
            n, a = ds.index[i]
            image, own, raw = ds.file['image'][n, a], int(ds.file['direction'][n, a]), ds.file['pixels'][n, a]
            k = kinds.numpy()
            fine = k * 100
            for r, c in zip(*np.nonzero((k == partner) | (k == ego))):
                tile = raw[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].reshape(-1, 3).astype(int)
                lit = tile[tile.sum(1) > 60]
                colour = int(lit.mean(0).argmax()) if len(lit) else 0
                direction = (int(image[c, r, 2]) - own) % 4 if k[r, c] == partner else 0   # image is (x, y)
                fine[r, c] += 10 * colour + direction
            return j, frame, kinds, torch.from_numpy(fine.astype(np.int16))

    return Frames()


def crop(frame, r, c, context=1):
    "The tile (r, c) of a (H, W, 3) frame with `context` tiles around it (black beyond the frame)."
    pad = np.pad(frame, ((context * TILE,) * 2, (context * TILE,) * 2, (0, 0)))
    return pad[r * TILE:(r + 1 + 2 * context) * TILE, c * TILE:(c + 1 + 2 * context) * TILE]


def counts_by(tokens, labels, values, K):
    "(K, len(values)): how many tiles of each label each code was given."
    col = {v: j for j, v in enumerate(values)}
    out = np.zeros((K, len(values)), np.int64)
    np.add.at(out, (tokens.ravel(), np.vectorize(col.get)(labels.ravel())), 1)
    return out


def compute(cfg, out: Path) -> None:
    "The tokens of `frames` validation frames, the counts, the galleries, the decodes; saved to tokens.npz and summary.json."
    import torch
    from hydra.utils import to_absolute_path
    from torch.utils.data import DataLoader
    from c3jepa_wm.data.datasets import TILE_NAMES, FindGoalPOVDataset
    from c3jepa_wm.data.transforms import get_transforms

    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    vqvae = hydra.utils.instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    vqvae.load_state_dict(torch.load(to_absolute_path(cfg.checkpoint), map_location='cpu')['model_state_dict'])
    vqvae = vqvae.to(device).eval()
    K, codebook = vqvae.num_embeddings, vqvae.vq_layer.embedding.detach()
    decode = lambda grid: vqvae.decode(codebook[grid].permute(0, 3, 1, 2))     # (B, 7, 7) codes -> frames

    ds = FindGoalPOVDataset(str(Path(cfg.data_dir) / 'dataset.h5'), 'val', tuple(cfg.val_layouts),
                            transform=get_transforms(img_size=224)[1])
    picks = np.sort(np.random.default_rng(cfg.seed).choice(len(ds), min(cfg.frames, len(ds)), replace=False))
    loader = DataLoader(frames_dataset(ds, picks), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    N, G = len(picks), cfg.gallery.examples
    tokens, kinds, fine = np.zeros((N, 7, 7), np.int16), np.zeros((N, 7, 7), np.uint8), np.zeros((N, 7, 7), np.int16)
    slot = {TILE_NAMES.index(n): i for i, n in enumerate(GALLERY_KINDS)}
    gallery = np.zeros((K, len(GALLERY_KINDS), G, 3 * TILE, 3 * TILE, 3), np.uint8)   # per code and kind, tiles of that kind it
    gallery_from = np.full((K, len(GALLERY_KINDS), G, 3), -1, np.int64)               # was given, in context; (frame, row, col)
    filled = np.zeros((K, len(GALLERY_KINDS)), int)
    errors = []                                                              # per frame, its tiles' largest pixel error
    with torch.no_grad():
        for j, x, k, f in loader:
            x = x.to(device)
            t = vqvae.get_message_indices(x)
            recon = decode(t)
            err = ((recon - x).abs() * 127.5).view(len(x), 3, 7, TILE, 7, TILE).amax(dim=(1, 3, 5))
            errors.append(err.cpu().numpy().astype(np.float16))
            t, j = t.cpu().numpy(), j.numpy()
            tokens[j], kinds[j], fine[j] = t, k.numpy(), f.numpy()
            images = to_uint8(x)
            kb = k.numpy()
            for b in range(len(t)):                                  # at most one tile per code and kind from a frame
                taken = set()
                for r, c in zip(*np.nonzero(np.isin(kb[b], list(slot)))):
                    code, g = int(t[b, r, c]), slot[int(kb[b, r, c])]
                    if (code, g) in taken or filled[code, g] == G:
                        continue
                    taken.add((code, g))
                    gallery[code, g, filled[code, g]] = crop(images[b], r, c)
                    gallery_from[code, g, filled[code, g]] = (j[b], r, c)
                    filled[code, g] += 1
            logger.info(f"{int(j[-1]) + 1} / {N} frames")
        errors = np.concatenate(errors)

        kind_values = sorted(int(v) for v in np.unique(kinds))
        fine_values = sorted(int(v) for v in np.unique(fine))
        counts_kind = counts_by(tokens, kinds, kind_values, K)
        counts_fine = counts_by(tokens, fine, fine_values, K)

        # each code alone, decoded
        single = np.concatenate([to_uint8(decode(torch.arange(s, min(s + 32, K), device=device)[:, None, None].expand(-1, 7, 7)))
                                 for s in range(0, K, 32)])

        # example frames: those showing the partner and the goal first, then the partner, then any
        partner, goal = TILE_NAMES.index('partner'), TILE_NAMES.index('goal')
        shows = lambda kind: (kinds == kind).any((1, 2))
        order = np.concatenate([np.nonzero(shows(partner) & shows(goal))[0], np.nonzero(shows(partner) & ~shows(goal))[0],
                                np.nonzero(~shows(partner))[0]])
        rng = np.random.default_rng(cfg.seed)
        chosen = [rng.permutation(part) for part in np.split(order, np.cumsum([(shows(partner) & shows(goal)).sum(),
                                                                                (shows(partner) & ~shows(goal)).sum()]))]
        examples = np.concatenate(chosen)[:cfg.examples]
        example_frames = np.stack([to_uint8(frames_dataset(ds, picks)[int(e)][1][None])[0] for e in examples])
        example_tokens = torch.as_tensor(tokens[examples].astype(np.int64), device=device)
        example_recon = to_uint8(decode(example_tokens))

        # swaps: one tile's token replaced by the main code of each label
        swap_codes = counts_fine.argmax(0)                                       # per label, the code it is given most
        r, c = cfg.swap_cell
        swaps = np.zeros((len(examples), len(fine_values), 224, 224, 3), np.uint8)
        for e in range(len(examples)):
            grids = example_tokens[e].repeat(len(fine_values), 1, 1)
            grids[:, r, c] = torch.as_tensor(swap_codes, device=device)
            swaps[e] = to_uint8(decode(grids))

        book = codebook.cpu().numpy()
        centred = book - book.mean(0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        pca = centred @ vt[:2].T

    np.savez_compressed(out / 'tokens.npz', picks=picks, tokens=tokens, kinds=kinds, fine=fine, tile_errors=errors,
                        counts_kind=counts_kind, counts_fine=counts_fine, gallery=gallery, gallery_from=gallery_from,
                        gallery_filled=filled, single=single, examples=examples, example_frames=example_frames,
                        example_recon=example_recon, swap_codes=swap_codes, swaps=swaps, codebook=book, pca=pca,
                        pca_variance=(sv[:2] ** 2) / (sv ** 2).sum())
    summary = dict(kinds=list(TILE_NAMES), kind_values=kind_values, fine_values=fine_values,
                   fine_names=[label_name(v, TILE_NAMES) for v in fine_values], swap_cell=list(cfg.swap_cell),
                   gallery_kinds=list(GALLERY_KINDS),
                   checkpoint=str(cfg.checkpoint), data_dir=str(cfg.data_dir), val_layouts=list(cfg.val_layouts),
                   frames=int(N), codes=int(K))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    logger.info(f"saved {out / 'tokens.npz'} and {out / 'summary.json'}")


# --------------------------------------------------------------------------------------------------- metrics

def entropy(p) -> float:
    "In bits."
    p = np.asarray(p, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def code_metrics(counts) -> dict:
    """Of a (codes, labels) table: the codes used, their purity (the share of tiles whose code's main label is theirs),
    H(label), H(label | code) and the share of H(label) the code tells."""
    used = counts.sum(1) > 0
    total = counts.sum()
    h = entropy(counts.sum(0))
    h_given = sum(counts[k].sum() / total * entropy(counts[k]) for k in np.nonzero(used)[0])
    return dict(codes_used=int(used.sum()), purity=float(counts.max(1).sum() / total), label_bits=h,
                label_bits_given_code=h_given, explained=1 - h_given / h if h > 0 else 1.0)


def codes_covering(column, share: float = 0.95) -> int:
    "How many codes give `share` of a label's tiles."
    c = np.sort(column)[::-1]
    return int(np.searchsorted(np.cumsum(c), share * c.sum()) + 1) if c.sum() else 0


def metrics(r, s) -> dict:
    "The scalars logged: per table its `code_metrics`, per label how many codes cover 95% of it, the tiles reconstructed."
    kind_names = [s['kinds'][v] for v in s['kind_values']]
    out = {f'kinds/{k}': v for k, v in code_metrics(r['counts_kind']).items()}
    out.update({f'labels/{k}': v for k, v in code_metrics(r['counts_fine']).items()})
    out.update({f'codes_for_95/{n}': codes_covering(r['counts_kind'][:, j]) for j, n in enumerate(kind_names)})
    out.update({f'codes_for_95/{n}': codes_covering(r['counts_fine'][:, j]) for j, n in enumerate(s['fine_names'])
                if n.startswith(('partner', 'ego'))})
    ok = r['tile_errors'].astype(np.float32) <= 64
    out.update({f'reconstructed/{n}': float(ok[r['kinds'] == v].mean()) for v, n in zip(s['kind_values'], kind_names)})
    return out


def code_table(r, s) -> tuple:
    "Per code: its usage, main kind and label, purity, and its count per kind (columns, rows)."
    kind_names = [s['kinds'][v] for v in s['kind_values']]
    ck, cf = r['counts_kind'], r['counts_fine']
    rows = []
    for k in np.nonzero(ck.sum(1))[0]:
        rows.append([int(k), int(ck[k].sum()), kind_names[ck[k].argmax()], s['fine_names'][cf[k].argmax()],
                     float(ck[k].max() / ck[k].sum()), *map(int, ck[k])])
    return ['code', 'tiles', 'main_kind', 'main_label', 'purity', *kind_names], rows


# --------------------------------------------------------------------------------------------------- figures

def code_order(counts_kind):
    "The used codes, grouped by their main kind (in the kinds' order), the most used first in each group."
    used = np.nonzero(counts_kind.sum(1))[0]
    main = counts_kind[used].argmax(1)
    return used[np.lexsort((-counts_kind[used].sum(1), main))], main[np.lexsort((-counts_kind[used].sum(1), main))]


def plot_kinds(r, s, path):
    "Each used code's share of each kind of tile (columns: codes by main kind), above the codes' usage."
    import matplotlib.pyplot as plt
    kind_names = [s['kinds'][v] for v in s['kind_values']]
    order, main = code_order(r['counts_kind'])
    counts = r['counts_kind'][order]
    share = counts / counts.sum(1, keepdims=True)
    fig, (top, ax) = plt.subplots(2, 1, figsize=(14, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
    top.bar(np.arange(len(order)), counts.sum(1), color=[KIND_COLOURS.get(kind_names[m], 'k') for m in main], width=1.0)
    top.set_yscale('log'), top.set_ylabel('tiles')
    ax.imshow(share.T, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(kind_names)), kind_names)
    for b in np.nonzero(np.diff(main))[0]:
        for a in (top, ax):
            a.axvline(b + 0.5, color='k', lw=0.6)
    starts = np.r_[0, np.nonzero(np.diff(main))[0] + 1]
    ax.set_xticks([st + (en - st) / 2 for st, en in zip(starts, np.r_[starts[1:], len(main)])],
                  [f'{kind_names[main[st]]} ({en - st})' for st, en in zip(starts, np.r_[starts[1:], len(main)])], fontsize=8)
    m = code_metrics(r['counts_kind'])
    top.set_title(f"{m['codes_used']} codes used; purity {m['purity']:.3f}; the code tells {100 * m['explained']:.1f}% "
                  f"of the kind's {m['label_bits']:.2f} bits")
    ax.set_xlabel('codes, grouped by their main kind (number of codes)')
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def plot_agents(r, s, path, share: float = 0.95):
    """For each label of the partner, the goal and the ego: the share of its tiles given each code, over the codes that
    together give `share` of some such label's tiles."""
    import matplotlib.pyplot as plt
    cols = [j for j, n in enumerate(s['fine_names']) if n.startswith(('partner', 'goal', 'ego'))]
    if not cols:
        return False
    counts = r['counts_fine'][:, cols]
    codes = set()
    for j in range(len(cols)):
        top = np.argsort(counts[:, j])[::-1]
        codes |= set(top[:codes_covering(counts[:, j], share)].tolist())
    codes = sorted(codes, key=lambda k: (counts[k].argmax(), -counts[k].max()))
    frac = counts[codes] / np.maximum(counts.sum(0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(cols), 1.0 + 0.22 * len(codes)))
    ax.imshow(frac, aspect='auto', cmap='Purples', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(cols)), [f"{s['fine_names'][j]}\n({counts[:, i].sum()})" for i, j in enumerate(cols)],
                  rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(codes)), [f'code {k}' for k in codes], fontsize=7)
    for i in range(len(codes)):
        for j in range(len(cols)):
            if frac[i, j] >= 0.01:
                ax.text(j, i, f'{100 * frac[i, j]:.0f}', ha='center', va='center', fontsize=6,
                        color='w' if frac[i, j] > 0.5 else 'k')
    ax.set_title(f"the codes of the partner, goal and ego tiles: % of each label's tiles\n(the codes giving {100 * share:g}% "
                 "of some label)", fontsize=9)
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)
    return True


def plot_gallery(r, s, kind, n_codes, path):
    """The codes one kind of tile is given most: per code, tiles of that kind it was given (the tile framed, a tile of
    context around it; the title: how many of the code's tiles are of this kind)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if kind not in s['gallery_kinds'] or s['kinds'].index(kind) not in s['kind_values']:
        return False
    g_kind = s['gallery_kinds'].index(kind)
    j = s['kind_values'].index(s['kinds'].index(kind))
    ck = r['counts_kind']
    mine = [k for k in np.argsort(ck[:, j])[::-1] if ck[k, j] > 0][:n_codes]
    G = r['gallery'].shape[2]
    fig, axes = plt.subplots(len(mine), G, figsize=(1.1 * G, 1.25 * len(mine)), squeeze=False)
    for i, k in enumerate(mine):
        for g in range(G):
            ax = axes[i, g]
            ax.set_axis_off()
            if g < r['gallery_filled'][k, g_kind]:
                ax.imshow(r['gallery'][k, g_kind, g])
                ax.add_patch(Rectangle((TILE - 0.5, TILE - 0.5), TILE, TILE, fill=False, ec='#ff00ff', lw=1))
        axes[i, 0].set_title(f"code {k}: {ck[k, j]} {kind} tiles of its {ck[k].sum()}", fontsize=7, loc='left')
    fig.suptitle(f"the {kind} tiles' main codes ({codes_covering(ck[:, j])} codes give 95% of them)", fontsize=9)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
    return True


def plot_single(r, s, path):
    "What each code draws on its own: the decoder on a 7x7 grid of it (title: its main kind; grey: unused)."
    import matplotlib.pyplot as plt
    kind_names = [s['kinds'][v] for v in s['kind_values']]
    ck = r['counts_kind']
    K = len(ck)
    n = int(np.ceil(np.sqrt(K)))
    fig, axes = plt.subplots(n, n, figsize=(n * 1.05, n * 1.15))
    for k, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if k < K:
            ax.imshow(r['single'][k])
            used = ck[k].sum() > 0
            ax.set_title(f"{k} {kind_names[ck[k].argmax()][:4] if used else '-'}", fontsize=6,
                         color=KIND_COLOURS.get(kind_names[ck[k].argmax()], 'k') if used else '#aaaaaa', pad=1)
    fig.suptitle('each code decoded alone (a 7x7 grid of it), titled with its main kind', fontsize=10)
    fig.tight_layout(), fig.savefig(path, dpi=110), plt.close(fig)


def plot_swaps(r, s, path):
    "Example frames with the token of `swap_cell` replaced by each label's main code, decoded (first column: no swap)."
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    E, L = r['swaps'].shape[:2]
    row, col = s['swap_cell']
    fig, axes = plt.subplots(E, L + 1, figsize=(1.35 * (L + 1), 1.45 * E), squeeze=False)
    for e in range(E):
        for i in range(L + 1):
            ax = axes[e, i]
            ax.set_axis_off()
            ax.imshow(r['example_recon'][e] if i == 0 else r['swaps'][e, i - 1])
            ax.add_patch(Rectangle((col * TILE - 0.5, row * TILE - 0.5), TILE, TILE, fill=False, ec='#ff00ff', lw=1))
            if e == 0:
                ax.set_title('reconstruction' if i == 0 else f"{s['fine_names'][i - 1]}\n(code {r['swap_codes'][i - 1]})",
                             fontsize=6)
    fig.suptitle(f'the token of tile {tuple(s["swap_cell"])} replaced by each label\'s main code', fontsize=9)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def plot_maps(r, s, path):
    "Example frames: the frame, its reconstruction, its 7x7 tokens (coloured by their main kind) and its tiles' kinds."
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    kinds = s['kinds']
    ck = r['counts_kind']
    kind_names = [kinds[v] for v in s['kind_values']]
    cmap = ListedColormap([KIND_COLOURS.get(n, '#000000') for n in kinds])
    E = len(r['examples'])
    fig, axes = plt.subplots(E, 4, figsize=(9, 2.3 * E), squeeze=False)
    for e, j in enumerate(r['examples']):
        t, k = r['tokens'][j], r['kinds'][j]
        main = np.vectorize(lambda c: kinds.index(kind_names[ck[c].argmax()]))(t)
        for ax, img, title in zip(axes[e], (r['example_frames'][e], r['example_recon'][e]), ('frame', 'reconstruction')):
            ax.imshow(img), ax.set_title(title, fontsize=8)
        axes[e, 2].imshow(main, cmap=cmap, vmin=0, vmax=len(kinds) - 1)
        axes[e, 2].set_title("tokens (coloured by the code's main kind)", fontsize=8)
        axes[e, 3].imshow(k, cmap=cmap, vmin=0, vmax=len(kinds) - 1)
        axes[e, 3].set_title('the tiles (kind)', fontsize=8)
        for rr in range(7):
            for cc in range(7):
                light = kinds[main[rr, cc]] in ('unseen', 'wall', 'goal')
                axes[e, 2].text(cc, rr, str(t[rr, cc]), ha='center', va='center', fontsize=6, color='k' if light else 'w')
                axes[e, 3].text(cc, rr, kinds[k[rr, cc]][:3], ha='center', va='center', fontsize=6,
                                color='k' if kinds[k[rr, cc]] in ('unseen', 'wall', 'goal') else 'w')
        for ax in axes[e]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def plot_codebook(r, s, path):
    "The codebook's 2-D PCA: each used code by its main kind, its size by its usage."
    import matplotlib.pyplot as plt
    kind_names = [s['kinds'][v] for v in s['kind_values']]
    ck, pca = r['counts_kind'], r['pca']
    used = ck.sum(1) > 0
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(*pca[~used].T, s=8, c='#cccccc', label='unused')
    main = ck.argmax(1)
    for j, n in enumerate(kind_names):
        m = used & (main == j)
        if m.any():
            ax.scatter(*pca[m].T, s=10 + 60 * np.sqrt(ck[m].sum(1) / ck.sum(1).max()), c=KIND_COLOURS.get(n, 'k'),
                       edgecolors='k', linewidths=0.3, label=f'{n} ({m.sum()})')
    v = r['pca_variance']
    ax.set_xlabel(f'PC 1 ({100 * v[0]:.0f}% of the variance)'), ax.set_ylabel(f'PC 2 ({100 * v[1]:.0f}%)')
    ax.legend(fontsize=7, loc='best'), ax.set_title("the codebook, by each code's main kind", fontsize=9)
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def plot(cfg, out: Path, run=None) -> None:
    "Every figure from <out_dir>'s saved results, into <out_dir>/figures (and to wandb, with the metrics and the code table)."
    r = dict(np.load(out / 'tokens.npz'))
    s = json.loads((out / 'summary.json').read_text())
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    drawn = {}
    plot_kinds(r, s, figures / 'kinds.png'), drawn.update(kinds=figures / 'kinds.png')
    if plot_agents(r, s, figures / 'agents.png'):
        drawn['agents'] = figures / 'agents.png'
    for kind in ('empty', 'wall', 'goal', 'partner', 'ego'):
        if plot_gallery(r, s, kind, cfg.gallery.codes, figures / f'gallery_{kind}.png'):
            drawn[f'gallery/{kind}'] = figures / f'gallery_{kind}.png'
    for name, f in (('codes', plot_single), ('swaps', plot_swaps), ('maps', plot_maps), ('codebook', plot_codebook)):
        f(r, s, figures / f'{name}.png')
        drawn[name] = figures / f'{name}.png'
    m = metrics(r, s)
    (out / 'metrics.json').write_text(json.dumps(m, indent=1))
    columns, rows = code_table(r, s)
    logger.info('\n' + '\n'.join(f'{k:40s} {v:.4g}' for k, v in m.items()))
    logger.info(f"figures in {figures}")
    if run is not None:
        import wandb
        run.log({**{f'tokens/{k}': v for k, v in m.items()},
                 **{f'tokens/{k}': wandb.Image(str(p)) for k, p in drawn.items()},
                 'tokens/codes': wandb.Table(columns=columns, data=rows)})


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/vq_tokens")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"tokens-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, job_type='analysis', dir=str(out),
                         config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    if cfg.stage in ('all', 'compute'):
        compute(cfg, out)
        if run is not None:                                  # the results themselves, to redraw from anywhere
            import wandb
            artifact = wandb.Artifact(f"vq-tokens-{os.getenv('SLURM_JOB_ID', 'local')}", type='analysis')
            for name in ('tokens.npz', 'summary.json'):
                artifact.add_file(str(out / name))
            run.log_artifact(artifact)
    if cfg.stage in ('all', 'plot'):
        plot(cfg, out, run)
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
