"""End-to-end text spotting evaluation (Table 1 protocol).

A predicted instance counts as correct (TP) iff:
  (1) its box matches a GT box with IoU >= 0.5, AND
  (2) its transcript equals the GT transcript (after normalization and, for the
      lexicon-constrained modes, lexicon correction of the predicted word).

Reports end-to-end Precision / Recall / F1 (Hmean) per dataset and per lexicon mode,
matching the columns of Table 1:
  Total-Text : None / Full
  ICDAR2015  : S(strong) / W(weak) / G(generic)
  CTW1500    : None / Full
  ATS        : None

Coordinates use pixel-space GT `solution` (same space GRPO trains the model to emit),
consistent with evaluation_det.py. Run from a dir containing gt.json and the lexicon
dirs (ic15/, ctw1500/, totaltext/). Reuses helpers from evaluation_rec / evaluation_det.
"""
import json, os, re
from collections import defaultdict
from tqdm import tqdm

# reuse the already-validated helpers
from evaluation_det import parse_predictions, iou, to_bbox, normalize_text as det_norm
from evaluation_rec import (normalize_text as rec_norm, load_all_vocabs,
                            get_vocab_for_mode, find_closest_word)

GT_FILE_PATH = os.environ.get("GT_FILE_PATH", "gt.json")
PRED_FILE_PATH = os.environ.get("PRED_FILE_PATH", "result.json")
TARGET_DATASETS = ['totaltext', 'ic15', 'ctw1500', 'ats']
IOU_THRESH = 0.5


def parse_gt_instances(gt_item):
    """GT boxes+text in pixel space from `solution`. Text normalized with det_norm to
    match parse_predictions() (which normalizes pred text the same way)."""
    out = []
    for sol in gt_item.get("solution", []):
        bbox = to_bbox(sol.get("bbox_2d") or sol.get("position"))
        if bbox:
            out.append({"bbox": bbox, "text": det_norm(str(sol.get("text_content", "")))})
    return out


def text_equal(pred_text, gt_text, vocab):
    """Transcript correctness, optionally after lexicon correction of the pred word."""
    if pred_text == gt_text:
        return True
    if vocab:
        corrected = det_norm(find_closest_word(pred_text, vocab))
        if corrected == gt_text:
            return True
    return False


def e2e_matches(gt_insts, pred_insts, vocab):
    """Greedy IoU matching (desc), TP requires IoU>=thresh AND transcript equal."""
    gt_used = [False] * len(gt_insts)
    pred_used = [False] * len(pred_insts)
    cand = []
    for pi, p in enumerate(pred_insts):
        for gi, g in enumerate(gt_insts):
            iou_v = iou(p['bbox'], g['bbox'])
            if iou_v >= IOU_THRESH:
                cand.append((iou_v, pi, gi))
    cand.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    for _, pi, gi in cand:
        if gt_used[gi] or pred_used[pi]:
            continue
        # box already matches; require transcript correct for E2E TP
        if text_equal(pred_insts[pi]['text'], gt_insts[gi]['text'], vocab):
            tp += 1
            gt_used[gi] = True
            pred_used[pi] = True
    fp = len(pred_insts) - sum(pred_used)
    fn = len(gt_insts) - sum(gt_used)
    return tp, fp, fn


def evaluate(gt_file, pred_file, target_datasets=None, ic_new=True):
    target_datasets = target_datasets or TARGET_DATASETS
    gt_all = json.load(open(gt_file, encoding='utf-8'))
    preds = []
    for line in open(pred_file, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    gt_filtered = [g for g in gt_all if g.get('dataset_name') in target_datasets]
    all_vocabs = load_all_vocabs(gt_filtered, ic_new)

    gt_map, basename2ds = {}, defaultdict(set)
    for g in gt_filtered:
        b = os.path.basename(g['image'])
        gt_map[(g['dataset_name'], b)] = g
        basename2ds[b].add(g['dataset_name'])

    pred_map = {}
    for p in preds:
        if not p.get('image'):
            continue
        b = os.path.basename(p['image'])
        ds = p.get('dataset_name')
        if not (ds and (ds, b) in gt_map):
            ds_set = list(basename2ds.get(b, []))
            ds = ds_set[0] if len(ds_set) == 1 else None
        if ds is not None:
            pred_map[(ds, b)] = parse_predictions(p.get('output', ''))

    results = {}
    for ds in target_datasets:
        keys = [k for k in gt_map if k[0] == ds]
        if not keys:
            continue
        if ds == 'ic15':
            modes = ['strong', 'weak', 'generic']
        elif ds in ('ctw1500', 'totaltext'):
            modes = ['none', 'full']
        else:
            modes = ['none']
        results[ds] = {}
        for mode in modes:
            tp = fp = fn = 0
            for (dsn, b) in keys:
                gt_insts = parse_gt_instances(gt_map[(dsn, b)])
                pred_insts = pred_map.get((dsn, b), [])
                vocab = get_vocab_for_mode(ds, mode, b, all_vocabs)
                a, c, d = e2e_matches(gt_insts, pred_insts, vocab)
                tp += a; fp += c; fn += d
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            results[ds][mode] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                                 'hmean': round(f1, 4), 'tp': tp, 'fp': fp, 'fn': fn}
    return results


if __name__ == '__main__':
    if not (os.path.exists(GT_FILE_PATH) and os.path.exists(PRED_FILE_PATH)):
        print(f"Error: need {GT_FILE_PATH} and {PRED_FILE_PATH}")
    else:
        res = evaluate(GT_FILE_PATH, PRED_FILE_PATH, ic_new=True)
        print("\n" + "=" * 60 + "\n  END-TO-END SPOTTING (Table 1 protocol, IoU>=0.5)\n" + "=" * 60)
        for ds in TARGET_DATASETS:
            if ds not in res:
                continue
            print(f"\n--- {ds.upper()} ---")
            for mode, m in res[ds].items():
                print(f"  {mode:>8}: Hmean={m['hmean']*100:.1f}  P={m['precision']*100:.1f} "
                      f"R={m['recall']*100:.1f}  (TP{m['tp']} FP{m['fp']} FN{m['fn']})")
