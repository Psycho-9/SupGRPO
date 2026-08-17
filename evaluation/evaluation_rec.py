import json
import re
import codecs
import os
from collections import defaultdict
from tqdm import tqdm

TARGET_DATASETS = ['ctw1500', 'ic15', 'totaltext', 'ats']
GT_FILE_PATH = os.environ.get("GT_FILE_PATH", "gt.json")
PRED_FILE_PATH = os.environ.get("PRED_FILE_PATH", "result.json")

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None


def unescape_string(s):
    try:
        return codecs.decode(s, 'unicode_escape')
    except Exception:
        return s


def normalize_text(text, strip_punctuation=False):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    if strip_punctuation:
        text = re.sub(r'[^a-z0-9]', '', text)
    return text


def edit_distance(s1, s2):
    if levenshtein_distance:
        return levenshtein_distance(s1, s2)
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions, deletions = previous_row[j + 1] + 1, current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def preprocess_vocab(vocab_list):
    return [(normalize_text(word, strip_punctuation=True), word) for word in vocab_list if word]


def find_closest_word(word, vocab_normed, max_distance=2):
    if not vocab_normed or not word:
        return word
    word_norm = normalize_text(word, strip_punctuation=True)
    if not word_norm:
        return word
    for norm_word, orig_word in vocab_normed:
        if word_norm == norm_word:
            return orig_word
    closest, min_dist = None, float('inf')
    for norm_word, orig_word in vocab_normed:
        dist = edit_distance(word_norm, norm_word)
        if dist < min_dist:
            min_dist, closest = dist, orig_word
    threshold = min(max_distance, max(1, len(word_norm) // 3))
    return closest if min_dist <= threshold else word


def apply_lexicon_correction(pred_words_raw, vocab, max_dist=2):
    if not vocab or not pred_words_raw:
        return pred_words_raw
    return [find_closest_word(word, vocab, max_distance=max_dist) for word in pred_words_raw]


def check_match_strict(gt_norm, pred_set_norm):
    if gt_norm in pred_set_norm:
        return True
    gt_tokens = set(gt_norm.split())
    if len(gt_tokens) > 1 and all(token in pred_set_norm for token in gt_tokens):
        return True
    pred_set_stripped = {normalize_text(t, strip_punctuation=True) for t in pred_set_norm}
    gt_stripped = normalize_text(gt_norm, strip_punctuation=True)
    if gt_stripped in pred_set_stripped:
        return True
    gt_stripped_tokens = set(gt_stripped.split())
    if len(gt_stripped_tokens) > 1 and all(token in pred_set_stripped for token in gt_stripped_tokens):
        return True
    return False


def check_match_flexible(gt_norm, pred_set_norm, vocab=None):
    if check_match_strict(gt_norm, pred_set_norm):
        return True
    pred_tokens_all = set(p for pred in pred_set_norm for p in pred.split())
    if vocab:
        pred_tokens_all = set(apply_lexicon_correction(list(pred_tokens_all), vocab))
    gt_tokens = gt_norm.split()
    if len(gt_tokens) > 1 and all(tok in pred_tokens_all for tok in gt_tokens):
        return True
    for p in pred_set_norm:
        if " " in p and p.replace(" ", "") == gt_norm:
            return True
    return False


def _json_candidates_from_output(output: str):
    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
    content = answer_match.group(1) if answer_match else output
    content = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", content.strip(), flags=re.DOTALL | re.IGNORECASE)
    candidates = [content]
    obj_start, obj_end = content.find("{"), content.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(content[obj_start:obj_end + 1])
    arr_start, arr_end = content.find("["), content.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(content[arr_start:arr_end + 1])
    return content, candidates


def _loads_json_candidate(json_str: str):
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str.strip())
    if not json_str:
        return None
    for candidate in (json_str, unescape_string(json_str)):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _collect_text_fields(obj, out):
    text_keys = ("text_content", "text", "word", "content")
    if isinstance(obj, dict):
        chosen_key = None
        for key in text_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
                chosen_key = key
                break
        for key, value in obj.items():
            if key == chosen_key:
                continue
            if key in text_keys and isinstance(value, str):
                continue
            _collect_text_fields(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_text_fields(item, out)


def extract_text_from_json_content(output: str):
    if not output or not output.strip():
        return []
    content, candidates = _json_candidates_from_output(output)
    seen = set()
    for json_str in candidates:
        if json_str in seen:
            continue
        seen.add(json_str)
        data = _loads_json_candidate(json_str)
        if data is None:
            continue
        words = []
        _collect_text_fields(data, words)
        if words:
            return words

    pattern = r"\\?\"(?:text_content|text|word|content)\\?\"\s*:\s*\\?\"((?:\\.|[^\"\\])*)\\?\""
    return [unescape_string(m) for m in re.findall(pattern, content)]


def load_all_vocabs(gt_data, ic_new):
    vocabs = {'ic15': {'strong': {}, 'weak': [], 'generic': []}}
    strong_dir, weak_path, gen_path = (
        "ic15/new_strong_lexicon", "ic15/ch4_test_vocabulary_new.txt", "ic15/GenericVocabulary_new.txt") if ic_new else (
        "ic15/strong_lexicon", "ic15/ch4_test_vocabulary.txt", "ic15/GenericVocabulary.txt")
    if os.path.exists(strong_dir):
        for item in (i for i in gt_data if i.get('dataset_name') == 'ic15'):
            match = re.search(r'(\d+)', os.path.basename(item['image']))
            if not match:
                continue
            lex_file = os.path.join(strong_dir,
                                    f"new_voc_img_{match.group(1)}.txt" if ic_new else f"voc_img_{match.group(1)}.txt")
            if os.path.exists(lex_file):
                with open(lex_file, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                    img_key = normalize_text(os.path.splitext(os.path.basename(item['image']))[0])
                    vocabs['ic15']['strong'][img_key] = preprocess_vocab(words)
    for path, key in [(weak_path, 'weak'), (gen_path, 'generic')]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                vocabs['ic15'][key] = preprocess_vocab([l.strip() for l in f if l.strip()])
    for ds in ['ctw1500', 'totaltext']:
        if os.path.exists(f"{ds}/weak_voc_new.txt"):
            with open(f"{ds}/weak_voc_new.txt", 'r', encoding='utf-8') as f:
                vocabs[ds] = {'full': preprocess_vocab([l.strip() for l in f if l.strip()])}
        else:
            vocabs[ds] = {'full': None}
    return vocabs


def get_vocab_for_mode(dataset, mode, img_name, all_vocabs):
    if mode == 'none':
        return None
    img_key = normalize_text(os.path.splitext(os.path.basename(img_name))[0])
    if dataset == 'ic15':
        return all_vocabs['ic15']['strong'].get(img_key) if mode == 'strong' else all_vocabs['ic15'].get(mode)
    if dataset in ['ctw1500', 'totaltext'] and mode == 'full':
        return all_vocabs[dataset].get('full')
    return None


def evaluate(gt_file, pred_file, ic_new=True, target_datasets=None):
    if not target_datasets:
        target_datasets = TARGET_DATASETS
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data_all = json.load(f)
    except Exception:
        return {}
    pred_data_all = []
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        pred_data_all.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return {}

    gt_valid_images = {os.path.basename(i['image']) for i in gt_data_all if i.get("normalized_solution")}
    pred_valid_images = {os.path.basename(p['image']) for p in pred_data_all if p.get("output", "").strip()}
    images_to_evaluate = gt_valid_images.intersection(pred_valid_images)

    gt_data_filtered = [i for i in gt_data_all if
                        os.path.basename(i['image']) in images_to_evaluate and i.get('dataset_name') in target_datasets]
    all_vocabs = load_all_vocabs(gt_data_filtered, ic_new)

    gt_map = defaultdict(dict)
    for item in gt_data_filtered:
        gt_map[item.get('dataset_name')][os.path.basename(item['image'])] = item

    pred_map = defaultdict(dict)
    for pred in pred_data_all:
        image_path = pred.get("image", "")
        if not image_path:
            continue
        try:
            parts = image_path.strip('/').split('/')
            dataset_name = parts[-3]
            img_basename = parts[-1]
            if dataset_name in target_datasets:
                pred_map[dataset_name][img_basename] = pred
        except IndexError:
            continue

    final_results = {}
    all_imgs_to_eval = [(ds, name) for ds, imgs in gt_map.items() for name in imgs]

    with tqdm(total=len(all_imgs_to_eval), desc="Evaluating") as pbar:
        for dataset, gt_imgs in gt_map.items():
            if dataset == 'ic15':
                modes = ['generic', 'weak', 'strong']
            elif dataset in ['ctw1500', 'totaltext']:
                modes = ['none', 'full']
            else:
                modes = ['none']

            rec_stats = defaultdict(lambda: defaultdict(int))
            for img_name, gt in gt_imgs.items():
                pbar.update(1)
                gt_solutions = gt.get("normalized_solution", [])
                pred = pred_map.get(dataset, {}).get(img_name)
                pred_words_raw = extract_text_from_json_content(pred['output']) if pred else []
                pred_set_norm = {normalize_text(t) for t in pred_words_raw if t}
                gt_texts_norm = [normalize_text(s['text_content']) for s in gt_solutions if s and s.get('text_content')]
                gt_set_norm = set(gt_texts_norm)
                directly_matched_preds_count = len(gt_set_norm.intersection(pred_set_norm))

                # multiset (bag-of-words) counts for word-level F1 (paper Eq. 5 / Table 2 metric)
                from collections import Counter
                gt_counter = Counter(gt_texts_norm)

                for mode in modes:
                    rec_stats[mode]['total'] += len(gt_texts_norm)
                    rec_stats[mode]['total_predictions'] += len(pred_words_raw)
                    rec_stats[mode]['directly_matched_predictions'] += directly_matched_preds_count
                    vocab = get_vocab_for_mode(dataset, mode, img_name, all_vocabs)
                    pred_set_corrected = set()
                    pred_words_for_f1 = pred_words_raw
                    if mode != 'none' and vocab:
                        remaining_exact = gt_counter.copy()
                        corrected_words = []
                        for word in pred_words_raw:
                            normalized_word = normalize_text(word)
                            if remaining_exact.get(normalized_word, 0) > 0:
                                corrected_words.append(word)
                                remaining_exact[normalized_word] -= 1
                            else:
                                corrected_words.append(find_closest_word(word, vocab))
                        pred_set_corrected = {normalize_text(t) for t in corrected_words if t}
                        pred_words_for_f1 = corrected_words

                    for gt_norm in gt_texts_norm:
                        is_matched = check_match_flexible(gt_norm, pred_set_norm, vocab=vocab if pred_set_corrected else None)
                        if is_matched:
                            rec_stats[mode]['matched'] += 1

                    # word-level multiset F1: TP(w)=min(count_pred, count_gt); aligns with
                    # paper's Rtext (Eq. 5) and the content_reward used in training.
                    pred_counter = Counter(normalize_text(t) for t in pred_words_for_f1 if t)
                    tp_multiset = sum(min(c, gt_counter.get(w, 0)) for w, c in pred_counter.items())
                    rec_stats[mode]['f1_tp'] += tp_multiset
                    rec_stats[mode]['f1_pred'] += sum(pred_counter.values())
                    rec_stats[mode]['f1_gt'] += sum(gt_counter.values())

            dataset_results = {}
            for mode in modes:
                matched, total = rec_stats[mode]['matched'], rec_stats[mode]['total']
                accuracy = matched / total if total > 0 else 0.0
                tp, npred, ngt = rec_stats[mode]['f1_tp'], rec_stats[mode]['f1_pred'], rec_stats[mode]['f1_gt']
                wp = tp / npred if npred > 0 else 0.0
                wr = tp / ngt if ngt > 0 else 0.0
                wf1 = 2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0.0
                dataset_results[mode] = {
                    'accuracy': round(accuracy, 4),
                    'word_f1': round(wf1, 4),
                    'word_precision': round(wp, 4),
                    'word_recall': round(wr, 4),
                    'matched': matched,
                    'total': total,
                    'total_predictions': rec_stats[mode]['total_predictions'],
                    'directly_matched_predictions': rec_stats[mode]['directly_matched_predictions']
                }
            final_results[dataset] = {'Recognition': dataset_results}

    return final_results


if __name__ == '__main__':
    if not os.path.exists(GT_FILE_PATH) or not os.path.exists(PRED_FILE_PATH):
        print(f"Error: Ensure '{GT_FILE_PATH}' and '{PRED_FILE_PATH}' exist.")
    else:
        results = evaluate(GT_FILE_PATH, PRED_FILE_PATH, ic_new=True, target_datasets=TARGET_DATASETS)
        print("\n\n" + "=" * 80 + "\n" + " " * 20 + "OVERALL EVALUATION RESULTS\n" + "=" * 80)
        for dataset in sorted(results.keys()):
            print(f"\n--- DATASET: {dataset.upper()} ---")
            if 'Recognition' in results[dataset]:
                print("  [Recognition]")
                for mode, res in results[dataset]['Recognition'].items():
                    print(
                        f"    - Mode '{mode:>7}': word-F1 = {res['word_f1']*100:.1f}  "
                        f"(P={res['word_precision']*100:.1f} R={res['word_recall']*100:.1f})  "
                        f"| flex-acc={res['accuracy']*100:.1f} ({res['matched']}/{res['total']})")
