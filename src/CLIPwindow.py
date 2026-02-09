from pathlib import Path
import torch
import matplotlib.pyplot as plt
import open_clip
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT


def build_clip(device, model_name="ViT-B-16", pretrained="openai"):
    """
    Returns (clip_model, preprocess, tokenizer).

    Tokenizer mismatch note:
      Use ONLY the tokenizer returned by open_clip.get_tokenizer(model_name) with this open_clip model.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_text_embeddings_openclip(clip_model, tokenizer, concept_texts, device, batch_size=256):
    all_embs = []
    for start in range(0, len(concept_texts), batch_size):
        batch = concept_texts[start:start + batch_size]
        tokens = tokenizer(batch).to(device)
        e = clip_model.encode_text(tokens)
        e = e / (e.norm(dim=1, keepdim=True) + 1e-12)
        all_embs.append(e)
    if not all_embs:
        return torch.empty(0, device=device)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def contrast_vector_delta(E_pos, E_neg):
    mu_pos, mu_neg = E_pos.mean(dim=0), E_neg.mean(dim=0)
    delta = mu_pos - mu_neg
    delta_norm = float(delta.norm().detach().cpu().item())
    delta = delta / (delta.norm() + 1e-12)
    return delta, delta_norm


def _dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def load_concepts_file(path, add_leading_space=True, lowercase=False, dedupe=True):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    concepts = []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        if lowercase:
            s = s.lower()
        if add_leading_space:
            s = " " + s
        concepts.append(s)

    return _dedupe_preserve_order(concepts) if dedupe else concepts


def load_concept_banks(data_dir, filenames, add_leading_space=True, lowercase=False, dedupe=True):
    data_dir = Path(data_dir)
    banks = {}
    for fn in filenames:
        print(f"Loading concept bank from {data_dir / fn}")
        banks[fn] = load_concepts_file(data_dir / fn, add_leading_space=add_leading_space,
                                       lowercase=lowercase, dedupe=dedupe)
    return banks


@torch.no_grad()
def build_text_embedding_banks(clip_model, tokenizer, concept_banks, device,batch_size=256):
    """
    Builds CLIP text embeddings for multiple concept banks.
    Embeddings are computed every time; no caching is performed.
    """
    T_banks = {}

    for bank_name, concepts in concept_banks.items():
        print(f"Computing text embeddings for bank {bank_name}")

        all_embs = []
        for start in range(0, len(concepts), batch_size):
            batch = concepts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)

            e = clip_model.encode_text(tokens)
            e = e / (e.norm(dim=1, keepdim=True) + 1e-12)
            all_embs.append(e)

        if not all_embs:
            T = torch.empty(0, device=device)
        else:
            T = torch.cat(all_embs, dim=0)

        T_banks[bank_name] = T

    return T_banks



@torch.no_grad()
def top_concepts_for_delta(T_text, concept_texts, delta, top_k=2):
    if T_text.numel() == 0 or not concept_texts:
        print("T_text.numel() == 0 or concept_texts is empty")
        quit()
    scores = T_text @ delta
    vals, idx = torch.topk(scores, k=top_k, largest=True)
    out = []
    for v, i in zip(vals, idx):
        out.append((concept_texts[int(i)], float(v.detach().cpu().item())))
    return out


@torch.no_grad()
def top_concepts_across_banks(T_banks, concept_banks, delta, top_k=2):
    out = {}
    for bank_name, T in T_banks.items():
        out[bank_name] = top_concepts_for_delta(T, concept_banks[bank_name], delta, top_k=top_k)

    # whats the best bank? sanity check
    best_bank, best_score = None, -1e9
    for bank_name, lst in out.items():
        if not lst:
            continue
        if lst[0][1] > best_score:
            best_score = lst[0][1]
            best_bank = bank_name
    print("Best bank:", best_bank, "best score:", best_score)
    return out


def pick_best_bank(top_by_bank):
    best_bank, best_score = None, -1e9
    for bank_name, lst in top_by_bank.items():
        if not lst:
            continue
        if lst[0][1] > best_score:
            best_score, best_bank = lst[0][1], bank_name
    return (None, []) if best_bank is None else (best_bank, top_by_bank[best_bank])


@torch.no_grad()
def window_top_bottom_k_indices(layer, window_size, top_k, device, step=1, max_windows=None):
    l = layer if isinstance(layer, torch.Tensor) else torch.as_tensor(layer)
    l = l.to(device)
    n_samples, K = l.shape

    starts = list(range(0, K - window_size + 1, step))
    if max_windows is not None:
        starts = starts[:int(max_windows)]

    top_list, bot_list, win_ranges = [], [], []
    top_unique, bot_unique = set(), set()
    top_total, bot_total = 0, 0

    for s in starts:
        e = s + window_size
        scores = l[:, s:e].mean(dim=1)

        top_idx = torch.topk(scores, k=top_k, largest=True).indices
        bot_idx = torch.topk(scores, k=top_k, largest=False).indices

        top_idx_list = [int(i) for i in top_idx.detach().cpu().tolist()]
        bot_idx_list = [int(i) for i in bot_idx.detach().cpu().tolist()]

        top_list.append(top_idx_list)
        bot_list.append(bot_idx_list)
        win_ranges.append((s, e))

        top_total += len(top_idx_list)
        bot_total += len(bot_idx_list)
        top_unique.update(top_idx_list)
        bot_unique.update(bot_idx_list)

    print(f"Top samples:    total = {top_total}, unique = {len(top_unique)}")
    print(f"Bottom samples: total = {bot_total}, unique = {len(bot_unique)}")
    print(f"Overlap (top ∩ bottom): {len(top_unique & bot_unique)} samples")

    return top_list, bot_list, win_ranges


def label_and_plot_windows_for_layer(emb, layer, dss, ds_key, device, out_dir, out_name,
    concept_data_dir, concept_filenames, window_size=3, step=3, top_k_set=50, top_k_concepts=2,
    clip_model_name="ViT-B-32", clip_pretrained="openai", text_batch_size=256, max_windows=None, title=None):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _ , tokenizer = build_clip(device=device, model_name=clip_model_name, pretrained=clip_pretrained)

    concept_banks = load_concept_banks(concept_data_dir, concept_filenames)
    T_banks = build_text_embedding_banks(
        clip_model, tokenizer, concept_banks, device,
        batch_size=text_batch_size,
    )
    top_sets, bot_sets, win_ranges = window_top_bottom_k_indices(layer, window_size, top_k_set, device,
                                                                 step=step, max_windows=max_windows)
    n_windows = len(win_ranges)
    print(f"Plotting {n_windows} windows of size {window_size}")

    E_all = emb._corevds[ds_key]["embedding"]

    fig_w = max(6, n_windows * 1.2)
    fig, axes = plt.subplots(2, n_windows, figsize=(fig_w, 3.2), constrained_layout=True)
    if n_windows == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title if title is not None else f"Layer window size {window_size}", fontsize=12)

    def _trim(s, n=32):
        return s if len(s) <= n else s[:n - 1] + "…"

    for w in range(n_windows):
        pos_idx, neg_idx = top_sets[w], bot_sets[w]

        E_pos = torch.as_tensor(E_all[pos_idx]).to(device).float()
        E_neg = torch.as_tensor(E_all[neg_idx]).to(device).float()
        E_pos = E_pos / (E_pos.norm(dim=1, keepdim=True) + 1e-12)
        E_neg = E_neg / (E_neg.norm(dim=1, keepdim=True) + 1e-12)

        delta, delta_norm = contrast_vector_delta(E_pos, E_neg)
        top_by_bank = top_concepts_across_banks(T_banks, concept_banks, delta, top_k=top_k_concepts)

        # pick the top-2 labels
        flat = []
        for bank_name, lst in top_by_bank.items():
            for concept, score in lst:
                flat.append((concept, float(score), bank_name))
        flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)

        if len(flat_sorted) == 0:
            label_text = "(no labels)"
        else:
            first = flat_sorted[0]
            second = flat_sorted[1] if len(flat_sorted) > 1 else (("", float("nan"), ""))
            c1, sc1, b1 = first
            c2, sc2, b2 = second
            # show both labels and bank name
            b1s = Path(b1).stem
            b2s = Path(b2).stem
            label_text = (f"1) {_trim(c1)}\n"
                          f"2) {_trim(c2)}")

        # best-max and best-min
        max_item = dss._dss[ds_key][[pos_idx[0]]][0]
        min_item = dss._dss[ds_key][[neg_idx[0]]][0]

        max_img = max_item["image"]
        if max_img.dim() == 4:
            max_img = max_img.squeeze(0)
        max_img = max_img.permute(1, 2, 0).detach().cpu().numpy()

        min_img = min_item["image"]
        if min_img.dim() == 4:
            min_img = min_img.squeeze(0)
        min_img = min_img.permute(1, 2, 0).detach().cpu().numpy()

        ax_top, ax_bot = axes[0, w], axes[1, w]
        ax_top.imshow(max_img); ax_bot.imshow(min_img)
        ax_top.axis("off"); ax_bot.axis("off")

        s0, e0 = win_ranges[w]
        ax_top.set_title(f"[{s0}:{e0})\n{label_text}\nΔnorm {delta_norm:.3f}", fontsize=7)

    save_path = out_dir / out_name
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return str(save_path)


