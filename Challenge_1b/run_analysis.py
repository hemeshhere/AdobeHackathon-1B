#!/usr/bin/env python3
import fitz, json, re, os, argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

SECTION_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')
FORM_ITEM_RE    = re.compile(r'^\s*\d+\.\s')

def parse_args():
    p = argparse.ArgumentParser(description="Persona-Driven PDF Section Ranking")
    p.add_argument("--pdf-dir",       required=True, help="Folder containing input PDFs")
    p.add_argument("--config",        required=True, help="Path to challenge1b_input.json")
    p.add_argument("--output",        required=True, help="Path to write challenge1b_output.json")
    p.add_argument("--top-sections",  type=int, default=5, help="How many top sections to return")
    p.add_argument("--top-sentences", type=int, default=3, help="How many top sentences per section")
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def extract_page_sections(pdf_path):
    doc = fitz.open(pdf_path)
    secs = []
    for page in doc:
        text = page.get_text().strip()
        if not text:
            continue
        secs.append({
            "document": os.path.basename(pdf_path),
            "page":    page.number + 1,
            "title":   f"Page {page.number + 1}",
            "text":    text
        })
    return secs

def split_sentences(text):
    return [s for s in SECTION_SPLIT_RE.split(text) if s.strip()]

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    pdfdir = args.pdf_dir
    docs   = cfg.get("documents", [])
    persona= cfg.get("persona", "")
    job    = cfg.get("job", "")
    query  = f"{persona}. {job}"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(query, convert_to_tensor=True)

    all_secs = []
    for fn in docs:
        path = os.path.join(pdfdir, fn)
        if os.path.exists(path):
            all_secs.extend(extract_page_sections(path))

    texts = [s["text"] for s in all_secs]
    sec_embs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, sec_embs)[0]

    ranked = scores.argsort(descending=True).tolist()[:args.top_sections]
    out_secs = []
    for rank, idx in enumerate(ranked, start=1):
        sec = all_secs[idx]
        out = {
            "document":        sec["document"],
            "page_number":     sec["page"],
            "section_title":   sec["title"],
            "importance_rank": rank,
            "sub_section_analysis": []
        }
        sents = split_sentences(sec["text"])
        sent_embs = model.encode(sents, convert_to_tensor=True)
        sent_scores = util.cos_sim(q_emb, sent_embs)[0]
        top_sidx = sent_scores.argsort(descending=True).tolist()[:args.top_sentences]
        for si in top_sidx:
            out["sub_section_analysis"].append({
                "refined_text": sents[si],
                "page_number":  sec["page"]
            })
        out_secs.append(out)

    result = {
        "metadata": {
            "documents":       docs,
            "persona":         persona,
            "job_to_be_done":  job,
            "timestamp":       datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": out_secs
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
