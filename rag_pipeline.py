"""
INDECIMAL Mini RAG Pipeline
============================
Embedding Model : sentence-transformers/all-MiniLM-L6-v2  (local, free)
Vector Store    : FAISS (local, in-memory)
LLM             : claude-sonnet-4-20250514 via Anthropic API
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import anthropic

# ─── 1. DOCUMENTS ────────────────────────────────────────────────────────────

DOCUMENTS = {
    "packages": """
INDECIMAL Package Pricing (per sqft, incl. GST):
- Essential: ₹1,851/sqft
- Premier (Most Popular): ₹1,995/sqft
- Infinia: ₹2,250/sqft
- Pinnacle: ₹2,450/sqft

Steel (Fe 550/Fe 550D):
- Essential: Sunvik/Kamadhenu up to ₹68,000/MT
- Premier: JSW/Jindal Neo up to ₹74,000/MT
- Infinia: JSW/Jindal Panther up to ₹75,000/MT
- Pinnacle: TATA or equivalent up to ₹80,000/MT

Cement (43 grade surface & 53 grade core):
- Essential: Dalmia/Bharthi up to ₹370/bag
- Premier: Dalmia/Bharthi up to ₹370/bag
- Infinia: Birla Super/Ramco up to ₹390/bag
- Pinnacle: ACC/Ultratech/Ramco up to ₹400/bag

Block Work: Solid concrete blocks 6" external & 4" internal.
6" blocks: up to ₹40 (+/-₹3) per block; 4" blocks: up to ₹33 (+/-₹3) per block.
RCC Mix: M20 or M25 (as advised by structural engineer).
Ceiling Height: Floor-to-floor 10 ft across all packages.
""",

    "kitchen_bathroom": """
Kitchen Ceramic Wall Dado:
- Essential: up to ₹40/sqft | Premier: up to ₹60/sqft
- Infinia: up to ₹80/sqft  | Pinnacle: up to ₹90/sqft

Kitchen Sink (Futura or equivalent):
- Essential: up to ₹4,000 (single bowl)
- Premier: up to ₹7,000 (single bowl)
- Infinia: up to ₹9,000 (single bowl with drain board)
- Pinnacle: up to ₹10,000 (single bowl with drain board)

Kitchen Sink Faucet:
- Essential/Premier: Parryware/Hindware up to ₹2,000/₹2,500
- Infinia/Pinnacle: Jaquar/Essco up to ₹3,500/₹4,000

Bathroom Ceramic Wall Dado: Same as Kitchen across packages.

Bathroom Sanitary & CP Fittings (per 1000 sqft):
- Essential: up to ₹32,000 (Cera/Hindware)
- Premier: up to ₹50,000 (Parryware)
- Infinia: up to ₹70,000 (Jaquar/Essco)
- Pinnacle: up to ₹80,000 (Kohler)

CPVC Pipe: APL Apollo/Ashirwad across all; Infinia/Pinnacle also include Supreme/Astral.
""",

    "doors_windows_painting": """
Main Door (includes fittings & labour):
- Essential: Panelled door up to ₹20,000
- Premier: Teak door up to ₹30,000
- Infinia: Teak door up to ₹40,000
- Pinnacle: Teak door up to ₹50,000

Windows (3-track with mosquito mesh, wallet per sqft):
- Essential: Aluminium/UPVC up to ₹440/sqft
- Premier: UPVC up to ₹500/sqft (Lesso eiti or equivalent)
- Infinia: UPVC up to ₹600/sqft
- Pinnacle: UPVC up to ₹700/sqft

Interior Painting (2-coat JK Putty + 1-coat Primer + 2-coat Emulsion):
- Essential: Asian Paints Tractor Emulsion
- Premier: Asian Paints Tractor Emulsion Advanced
- Infinia: Asian Paints Tractor Emulsion Shyne
- Pinnacle: Asian Paints Royale Emulsion

Exterior Painting (1-coat Primer + 2-coat Exterior Emulsion):
- Essential: Asian Paints Ace Emulsion
- Premier: Asian Paints Ace Shyne Emulsion
- Infinia: Asian Paints Apex Emulsion
- Pinnacle: Asian Paints Apex Ultima Emulsion
""",

    "flooring": """
Living & Dining Flooring (wallet per sqft):
- Essential: tiles up to ₹50/sqft
- Premier: tiles/granite up to ₹100/sqft
- Infinia: tiles/granite/marble up to ₹140/sqft
- Pinnacle: tiles/granite/marble up to ₹170/sqft

Rooms & Kitchen Flooring (wallet per sqft):
- Essential: tiles up to ₹50/sqft
- Premier: tiles/granite up to ₹80/sqft
- Infinia: tiles up to ₹110/sqft
- Pinnacle: tiles up to ₹140/sqft

Note: Laying charges vary and are separate from material wallet amounts.
Wallet amounts are spending caps; upgrades can be customized at additional cost.
""",

    "quality_payments": """
Payment Safety & Stage Controls:
- Customer payments go to an escrow account.
- Project manager verifies stage completion before funds are released to the construction partner.
- Purpose: reduce financial risk and improve transparency.

Quality Assurance System — 445+ Critical Checkpoints:
- Covers the entire construction lifecycle.
- Each phase scored and audited for: structural integrity, safety compliance, execution accuracy.
- Metrics accessible via customer dashboard.

Delay Management — Zero-Tolerance Policy:
- Integrated project management system with daily tracking.
- Instant flagging of deviations and automated task assignment.
- Penalisation mechanism to reinforce on-time delivery.

Post-Construction Maintenance (Zero Cost Program):
Coverage includes: Plumbing, electrical, wardrobe, masonry, modular kitchen,
CP fittings, crack filling, tile support, roofing, painting, external windows & doors.
""",

    "company_journey": """
Indecimal — What We Do:
End-to-end home construction with transparent pricing, quality assurance, and structured tracking.

Core Commitments:
1. Smooth Construction Experience — step-by-step support throughout.
2. Best & Competitive Pricing — fair pricing with no hidden charges.
3. Quality Assurance — 445+ checks at every stage.
4. Stage-Based Contractor Payments — released only after verified completion.
5. Transparent & Live Tracking — real-time online project monitoring.

Key Differentiators vs market alternatives:
- Long-term warranty & post-delivery support.
- 100% transparent pricing and process.
- Fixed project timelines with penalties for delays.
- Branded materials and on-site quality checks.
- Real-time project tracking dashboard.

Customer Journey (10 stages):
1. Raise a request — share plot details & vision.
2. Meet our experts — architects & construction specialists.
3. Get home financing — documentation to disbursal guidance.
4. Design your custom home — collaborative architectural design.
5. Receive plans — detailed design + cost plans, transparent pricing.
6. Book with us — milestone-based payments with clear timelines.
7. Real-time construction progress — live photo updates via app/dashboard.
8. Interior — concept-to-completion interiors customization.
9. Move in — handover with maintenance & structural warranty.
10. Maintenance — post-handover long-term care support.

Financing Support: Dedicated relationship manager, minimal documentation,
confirmation ~7 days*, disbursal ~30 days* (eligibility dependent).

Dedicated Team Roles: Expert advisor, relationship manager, architect,
project manager, site engineer, interiors support, maintenance support.

Partner Onboarding: Project verification → background & financial check →
agreement signing (SOP) → onboarding across Bangalore.
""",
}

# ─── 2. CHUNKING ─────────────────────────────────────────────────────────────

def chunk_documents(docs: dict, chunk_size: int = 300, overlap: int = 50):
    """Split each document into overlapping text chunks."""
    chunks = []
    for section, text in docs.items():
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"section": section, "text": chunk_text})
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks

# ─── 3. EMBED & INDEX ────────────────────────────────────────────────────────

def build_index(chunks: list, model: SentenceTransformer):
    """Embed chunks and build FAISS index."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on L2-normalised = cosine
    index.add(embeddings)
    return index, embeddings

# ─── 4. RETRIEVE ─────────────────────────────────────────────────────────────

def retrieve(query: str, model: SentenceTransformer, index, chunks: list, top_k: int = 3):
    """Return top-k chunks most relevant to query."""
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "section": chunks[idx]["section"],
            "text": chunks[idx]["text"],
            "score": float(score),
        })
    return results

# ─── 5. GENERATE ─────────────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list, client: anthropic.Anthropic) -> str:
    """Generate a grounded answer using only retrieved context."""
    context = "\n\n---\n\n".join(
        f"[Section: {c['section']}]\n{c['text']}" for c in retrieved_chunks
    )
    system_prompt = """You are the INDECIMAL AI Assistant — a helpful, precise assistant for an Indian home construction marketplace.

STRICT RULES:
1. Answer ONLY using the provided context below. Do NOT use external knowledge.
2. If the context does not contain enough information, say so clearly.
3. Quote specific numbers, brand names, and allowances from the context when relevant.
4. Keep answers concise, structured, and easy to read.
5. Do NOT speculate or invent any figures or policies."""

    user_prompt = f"""CONTEXT (retrieved from internal documents):
{context}

USER QUESTION: {query}

Answer strictly based on the context above:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text

# ─── 6. FULL PIPELINE ────────────────────────────────────────────────────────

class IndecimalRAG:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = anthropic.Anthropic()
        self.chunks = chunk_documents(DOCUMENTS)
        print(f"Created {len(self.chunks)} chunks from {len(DOCUMENTS)} sections.")
        self.index, _ = build_index(self.chunks, self.embed_model)
        print("FAISS index built.")

    def query(self, question: str, top_k: int = 3) -> dict:
        retrieved = retrieve(question, self.embed_model, self.index, self.chunks, top_k)
        answer = generate_answer(question, retrieved, self.client)
        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = IndecimalRAG()
    test_questions = [
        "What is the price of the Premier package?",
        "Which steel brand is used in the Pinnacle package?",
        "What flooring options are available for living room in Infinia?",
        "How does Indecimal ensure quality during construction?",
        "What is the main door wallet for Essential package?",
        "How are contractor payments handled?",
        "What painting brand is used for exterior in Pinnacle?",
        "What sanitary fittings are provided in the Infinia package?",
    ]
    results = []
    for q in test_questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer'][:200]}...")
        results.append(result)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nEvaluation results saved to eval_results.json")
