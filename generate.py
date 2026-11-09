"""Grounded, cited answer generation with a relevance-gated refusal."""
from .text import detect_lang

REFUSAL = {
    "English": "I couldn't find an answer to this in the loaded documents.",
    "Arabic": "لم أجد إجابة عن هذا السؤال في المستندات المحمّلة.",
}

SYSTEM = """You are a bilingual (Arabic/English) assistant that answers questions about WHO documents.
Rules:
1. Answer ONLY from the numbered context passages. Do not use outside knowledge.
2. Reply in {lang}, even if the passages are in another language.
3. Put the citation number like [1] at the end of EACH sentence or list item it supports.
4. If the passages do not contain the answer, say so in one sentence and add NO citations.
Be concise."""


def build_messages(question: str, hits, chunks, lang: str):
    context = "\n\n".join(
        f"[{n}] ({chunks[i]['source']}, p.{chunks[i]['page']})\n{chunks[i]['text']}"
        for n, (i, _) in enumerate(hits, 1)
    )
    return [
        {"role": "system", "content": SYSTEM.format(lang=lang)},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def answer(question: str, retriever, generate_fn, min_score: float = 0.30,
           top_n: int = 4, n_candidates: int = 30):
    """Return (answer_text, hits). Refuses without calling the LLM if nothing is relevant."""
    lang = detect_lang(question)
    hits = [h for h in retriever.rerank(question, top_n=top_n, n_candidates=n_candidates)
            if h[1] >= min_score]
    if not hits:
        return REFUSAL[lang], []
    return generate_fn(build_messages(question, hits, retriever.chunks, lang)), hits


# ---- pluggable generators (messages -> text) ------------------------------
def make_local_generator(model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                         load_in_4bit: bool = True, max_new_tokens: int = 400):
    """Run an open model locally with transformers (GPU needed for 7B)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    llm = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def generate(messages):
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(llm.device)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def make_api_generator(model_id: str = "Qwen/Qwen2.5-7B-Instruct", token: str | None = None,
                       max_new_tokens: int = 400):
    """Call a hosted model through Hugging Face Inference Providers."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model_id, token=token)

    def generate(messages):
        resp = client.chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=0.1)
        return resp.choices[0].message.content

    return generate
