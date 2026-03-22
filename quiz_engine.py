from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_store import hash_text_to_vector, load_index, search_vector_index


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "llama2")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
QUESTION_RETRY_LIMIT = 2
SUBJECTS = ["History", "Polity", "Economy", "Geography", "Environment", "Science", "Current Affairs"]
BASE_DIR = Path(__file__).resolve().parent
NOTES_PATH = BASE_DIR / "upsc_research_notes.md"
RAG_INDEX_PATH = BASE_DIR / "rag_store" / "index.json"
RAG_VECTOR_INDEX_PATH = BASE_DIR / "rag_store" / "vector_index.json"
CURRENT_RAG_INDEX_PATH = BASE_DIR / "rag_store_current" / "index.json"
CURRENT_RAG_VECTOR_INDEX_PATH = BASE_DIR / "rag_store_current" / "vector_index.json"
SUBJECT_KEYWORDS = {
    "History": ["history", "modern india", "art and culture", "ancient", "bhakti", "spectrum", "ncert history", "heritage"],
    "Polity": ["polity", "constitution", "constitutional", "laxmikanth", "federalism", "parliament", "reservation act", "criminal laws", "dpdp"],
    "Economy": ["economy", "gst", "ibc", "upi", "inflation", "budget", "green hydrogen", "macro", "digital economy"],
    "Geography": ["geography", "g.c. leong", "river", "soil", "mapping", "climate", "monsoon", "wetlands"],
    "Environment": ["environment", "shankar", "biodiversity", "conservation", "wetlands", "climate", "mission life", "green transitions"],
    "Science": ["science", "science-tech", "chandrayaan", "aditya-l1", "indiaai", "semiconductors", "technology", "space"],
    "Current Affairs": ["current", "g20", "dpdp", "indiaai", "green hydrogen", "women's reservation", "criminal laws", "biofuel", "article 370"],
}


@dataclass(frozen=True)
class FactItem:
    subject: str
    prompt: str
    answer: str
    distractors: tuple[str, str, str]
    explanation: str
    reference: str


def _facts() -> dict[str, list[FactItem]]:
    return {
        "History": [
            FactItem("History", "Which Harappan site is best known for a dockyard-like structure?", "Lothal", ("Kalibangan", "Rakhigarhi", "Banawali"), "Lothal is widely identified with a dockyard-like structure and maritime trade.", "NCERT Ancient India / Indus Valley"),
            FactItem("History", "Ashokan inscriptions in Greek and Aramaic are associated with which location?", "Kandahar", ("Sarnath", "Dhauli", "Girnar"), "Kandahar inscriptions reflect Ashoka's communication with frontier regions.", "NCERT Ancient India / inscriptions"),
            FactItem("History", "The doctrine of Drain of Wealth is most closely associated with whom?", "Dadabhai Naoroji", ("M.G. Ranade", "W.C. Bonnerjee", "C. Rajagopalachari"), "Dadabhai Naoroji argued that colonial rule transferred wealth from India to Britain.", "Spectrum / economic nationalism"),
            FactItem("History", "The Ilbert Bill controversy highlighted what in colonial India?", "Racial discrimination in administration", ("Demand for dominion status", "Indigo peasant revolt", "Separate electorates"), "The controversy exposed the racial hierarchy of colonial rule.", "Spectrum / colonial administration"),
            FactItem("History", "Dyarchy under the Government of India Act, 1919 applied to which level?", "Provincial subjects", ("Princely states", "Union territories", "The judiciary"), "Provincial subjects were split into reserved and transferred categories.", "Spectrum / constitutional history"),
            FactItem("History", "Permanent Settlement fixed land revenue permanently with whom?", "Zamindars", ("Ryots directly", "Village panchayats", "Moneylenders"), "The Permanent Settlement recognized zamindars as proprietors and fixed revenue demand.", "Spectrum / land revenue"),
            FactItem("History", "Who is most associated with the Alvars tradition?", "Vaishnava poet-saints of Tamil region", ("Jain monks of Karnataka", "Sufi saints of Punjab", "Tantric teachers of Bengal"), "The Alvars were devotional Vaishnava saints in South India.", "Art and Culture / Bhakti"),
            FactItem("History", "Yakshagana is a traditional theatre form of which state?", "Karnataka", ("Odisha", "Rajasthan", "Tamil Nadu"), "Yakshagana developed as a dance-drama form in Karnataka.", "Art and Culture / state-art mapping"),
            FactItem("History", "The Poona Pact of 1932 was signed between Mahatma Gandhi and whom?", "B.R. Ambedkar", ("Muhammad Ali Jinnah", "Subhas Chandra Bose", "Madan Mohan Malaviya"), "The pact revised representation arrangements for depressed classes.", "Spectrum / constitutional developments"),
            FactItem("History", "The term 'Satyashodhak Samaj' is associated with whom?", "Jyotiba Phule", ("Swami Dayanand Saraswati", "Ishwar Chandra Vidyasagar", "Raja Rammohan Roy"), "Jyotiba Phule founded the Satyashodhak Samaj.", "Spectrum / social reform"),
        ],
        "Polity": [
            FactItem("Polity", "A Money Bill can be introduced only in which House?", "Lok Sabha", ("Rajya Sabha", "Either House", "A joint sitting"), "A Money Bill can be introduced only in the Lok Sabha.", "Laxmikanth / parliamentary procedure"),
            FactItem("Polity", "Which body recommends distribution of tax revenues between Union and States?", "Finance Commission", ("Election Commission", "Inter-State Council", "GST Council"), "The Finance Commission handles tax devolution recommendations.", "Laxmikanth / constitutional bodies"),
            FactItem("Polity", "Anti-defection provisions are contained in which Schedule?", "Tenth Schedule", ("Fifth Schedule", "Seventh Schedule", "Twelfth Schedule"), "The Tenth Schedule contains anti-defection provisions.", "Laxmikanth / schedules"),
            FactItem("Polity", "The 73rd Constitutional Amendment is associated with what?", "Panchayati Raj institutions", ("Municipal corporations", "Cooperative societies", "Emergency provisions"), "The 73rd Amendment constitutionalized Panchayati Raj.", "Laxmikanth / local governance"),
            FactItem("Polity", "The GST Council is chaired by whom?", "Union Finance Minister", ("Prime Minister", "Vice President", "Chief Justice of India"), "The Union Finance Minister chairs the GST Council.", "Polity / GST"),
            FactItem("Polity", "Which of the following is not a Fundamental Duty?", "Voting in every election", ("Protecting the environment", "Developing scientific temper", "Safeguarding public property"), "Voting is important civically, but it is not a Fundamental Duty.", "Laxmikanth / duties"),
            FactItem("Polity", "One-third of Rajya Sabha members retire after what interval?", "Every second year", ("Every year", "Every three years", "Every five years"), "Rajya Sabha is a permanent body with biennial retirement by one-third members.", "Laxmikanth / Parliament"),
            FactItem("Polity", "The Comptroller and Auditor General of India is best described as what?", "A constitutional authority auditing public finances", ("A statutory tribunal", "A department under the Finance Ministry", "A court of accounts"), "The CAG is central to public audit and accountability.", "Laxmikanth / accountability"),
            FactItem("Polity", "The Nari Shakti Vandan Adhiniyam, 2023 provides reservation for women in which bodies?", "Lok Sabha and State Legislative Assemblies", ("Rajya Sabha only", "All elected bodies including panchayats", "Legislative Councils only"), "The amendment provides one-third reservation in Lok Sabha and State Assemblies, subject to implementation conditions.", "Current Affairs / women's reservation"),
            FactItem("Polity", "Which article of the Constitution deals with constitutional remedies?", "Article 32", ("Article 14", "Article 19", "Article 368"), "Article 32 provides the right to constitutional remedies.", "Laxmikanth / fundamental rights"),
            FactItem("Polity", "Who decides whether a bill is a Money Bill in the Lok Sabha?", "The Speaker of the Lok Sabha", ("The President", "The Prime Minister", "The Rajya Sabha Chairperson"), "The Speaker's decision is final in classifying a Money Bill.", "Laxmikanth / Parliament"),
            FactItem("Polity", "Which emergency is provided under Article 356?", "President's Rule in a state", ("National Emergency due to war", "Financial Emergency", "Judicial Emergency"), "Article 356 deals with failure of constitutional machinery in a state.", "Laxmikanth / emergency provisions"),
        ],
        "Economy": [
            FactItem("Economy", "Primary deficit is equal to fiscal deficit minus what?", "Interest payments", ("Capital expenditure", "Tax revenue", "Subsidies"), "Primary deficit isolates borrowing net of interest burden.", "NCERT Economy / budget basics"),
            FactItem("Economy", "Repo rate is the rate at which RBI does what?", "Lends short-term funds to commercial banks", ("Borrows from households", "Lends to NBFC customers", "Issues government bonds"), "Repo is a key monetary policy rate used by RBI.", "Economy / monetary policy"),
            FactItem("Economy", "Stagflation means which combination?", "High inflation with weak growth and unemployment", ("Low inflation with high growth", "Deflation with booming exports", "Balanced budget with price stability"), "Stagflation combines inflation with stagnation.", "NCERT Economy / macro concepts"),
            FactItem("Economy", "MSP refers to what in Indian agriculture?", "Government-announced support price for selected crops", ("Retail price cap", "Minimum farm wage", "Crop insurance premium"), "MSP is announced to support selected crops.", "Economy / agriculture"),
            FactItem("Economy", "Capital expenditure by government typically does what?", "Creates assets or reduces liabilities", ("Pays only salaries", "Increases only subsidies", "Counts as a revenue receipt"), "Capital expenditure includes investment in assets and infrastructure.", "Economy / public finance"),
            FactItem("Economy", "UPI is best described as what?", "An interoperable instant payment system", ("A cryptocurrency", "A stock exchange platform", "A central bank digital currency"), "UPI is part of India's digital public infrastructure.", "Current Affairs / digital economy"),
            FactItem("Economy", "The Insolvency and Bankruptcy Code, 2016 created what kind of framework?", "Time-bound insolvency resolution", ("Universal basic income system", "Direct tax code", "Exchange rate band"), "IBC streamlined insolvency resolution.", "Current Affairs / reforms"),
            FactItem("Economy", "Inflation targeting in India is formally linked with which institution?", "Monetary Policy Committee", ("Finance Commission", "NITI Aayog", "Public Accounts Committee"), "The MPC sets the policy repo rate under inflation targeting.", "Economy / RBI"),
            FactItem("Economy", "Which index is most directly used for retail inflation targeting in India?", "Consumer Price Index", ("Index of Industrial Production", "Wholesale Trade Index", "Human Development Index"), "Retail inflation targeting uses CPI.", "Economy / inflation data"),
            FactItem("Economy", "Green hydrogen is generally produced through electrolysis powered by what?", "Renewable energy", ("Diesel generators", "Coal gasification alone", "Nuclear weapons complexes"), "Green hydrogen emphasizes low-emission production using renewable electricity.", "Current Affairs / energy transition"),
            FactItem("Economy", "A rise in CRR usually has what effect on bank lending capacity?", "It reduces lending capacity", ("It guarantees export growth", "It increases gold imports", "It abolishes inflation"), "Higher CRR lowers lendable resources with banks.", "Economy / monetary tools"),
        ],
        "Geography": [
            FactItem("Geography", "The Inter-Tropical Convergence Zone is where which winds converge?", "Northeast and southeast trade winds", ("Polar easterlies", "Westerlies and monsoon winds", "Jet streams"), "The ITCZ is a low-pressure zone where trade winds converge.", "NCERT Geography / climatology"),
            FactItem("Geography", "Black soil in India is mainly associated with which region type?", "Basaltic lava regions of the Deccan", ("Young fold mountains", "Deltaic sands", "Coral island surfaces"), "Black soil derives largely from basalt weathering.", "NCERT Geography / soils"),
            FactItem("Geography", "Which major river is west-flowing?", "Narmada", ("Godavari", "Krishna", "Mahanadi"), "Narmada flows westward into the Arabian Sea.", "NCERT Geography / Indian rivers"),
            FactItem("Geography", "Western disturbances are mainly what?", "Temperate cyclones from the Mediterranean region", ("Equatorial easterlies", "Tropical anticyclones", "Local land breezes"), "Western disturbances bring winter rainfall to northwestern India.", "NCERT Geography / climate"),
            FactItem("Geography", "The standard meridian of India is based on which longitude?", "82.5 degree East", ("68 degree East", "75 degree East", "90 degree East"), "82.5 degree East is used for Indian Standard Time.", "NCERT Geography / coordinates"),
            FactItem("Geography", "Laterite soils generally develop under which conditions?", "High temperature and heavy rainfall", ("Glacial erosion", "Low evaporation deserts", "Marine coral growth"), "Intense leaching under tropical wet conditions leads to laterite soil formation.", "NCERT Geography / soils"),
            FactItem("Geography", "Which mountain pass connects Kashmir with Ladakh?", "Zoji La", ("Shipki La", "Nathu La", "Bomdi La"), "Zoji La is a strategic pass in the western Himalayas.", "NCERT Geography / passes"),
            FactItem("Geography", "A rain shadow region develops on which side of a mountain barrier?", "Leeward side", ("Windward side", "Delta side", "Polar side"), "Descending air on the leeward side suppresses rainfall.", "Physical Geography / climatology"),
            FactItem("Geography", "The Tropic of Cancer passes through how many Indian states?", "Eight", ("Six", "Seven", "Nine"), "The Tropic of Cancer crosses eight Indian states.", "NCERT Geography / mapping"),
            FactItem("Geography", "Which one of the following is a tributary of the Brahmaputra?", "Subansiri", ("Betwa", "Periyar", "Sabarmati"), "Subansiri is an important tributary of the Brahmaputra.", "NCERT Geography / river systems"),
        ],
        "Environment": [
            FactItem("Environment", "Mangroves are important because they do what?", "Protect coasts and nurture marine life", ("Grow only in glaciers", "Need zero salinity", "Occur only in deep deserts"), "Mangroves stabilize shorelines and support marine nurseries.", "Environment / coastal ecology"),
            FactItem("Environment", "Which is a biodiversity hotspot partly located in India?", "Western Ghats-Sri Lanka", ("Gobi Desert", "Patagonian Steppe", "Siberian Taiga"), "The Western Ghats-Sri Lanka hotspot is globally recognized for high endemism.", "Environment / hotspots"),
            FactItem("Environment", "A Ramsar site is what?", "A wetland of international importance", ("A tiger reserve", "A coal-bearing basin", "A desert biosphere zone"), "Ramsar designation recognizes wetlands under the Ramsar Convention.", "Environment / conventions"),
            FactItem("Environment", "An ecotone is best described as what?", "A transition zone between ecosystems", ("A cold ocean trench", "A volcanic plateau", "A permanent snowfield"), "Ecotones show edge effects and mixed ecological characteristics.", "Environment / ecology"),
            FactItem("Environment", "Which is an example of in-situ conservation?", "National park", ("Seed bank", "Cryopreservation unit", "Botanical garden"), "In-situ conservation protects species in their natural habitats.", "Environment / conservation"),
            FactItem("Environment", "Endemic species are species that are what?", "Restricted to a specific geographic area", ("Found on every continent", "Always invasive", "Always domesticated"), "Endemic species have limited natural distribution.", "Environment / species concepts"),
            FactItem("Environment", "The Montreal Protocol is chiefly associated with protection of what?", "Ozone layer", ("Tropical forests", "Whale sanctuaries", "World heritage monuments"), "The Montreal Protocol targets ozone-depleting substances.", "Environment / global conventions"),
            FactItem("Environment", "Mission LiFE is associated with what broad idea?", "Environmentally sustainable lifestyles", ("A crewed lunar mission", "Universal crop insurance", "New tax code"), "Mission LiFE promotes mindful and sustainable behavior change.", "Current Affairs / climate action"),
            FactItem("Environment", "Coral bleaching is most directly linked with what stress?", "Rise in sea surface temperature", ("Increase in soil salinity", "Decline in groundwater table", "Wind erosion"), "Heat stress can disrupt coral symbiosis and cause bleaching.", "Environment / marine ecology"),
        ],
        "Science": [
            FactItem("Science", "Chandrayaan-3 is especially noted for what achievement?", "Soft landing near the lunar south polar region", ("First human moonwalk by India", "Lunar sample return", "Permanent moon station"), "Chandrayaan-3 demonstrated India's successful soft landing capability near the Moon's south polar region.", "Current Affairs / space missions"),
            FactItem("Science", "Aditya-L1 is designed mainly to study what?", "The Sun from the Sun-Earth L1 point", ("The Earth's core", "Asteroids in the Kuiper Belt", "Deep-sea vents"), "Aditya-L1 is India's solar observatory mission.", "Current Affairs / solar mission"),
            FactItem("Science", "The Sun-Earth L1 point is useful because it allows what?", "Continuous observation of the Sun", ("Direct access to the Moon", "Operation inside the Earth's atmosphere", "Geostationary coverage over India"), "A spacecraft near L1 can monitor the Sun with minimal interruption by Earth.", "Science / astronomy basics"),
            FactItem("Science", "Vaccines generally work by doing what?", "Training the immune system to recognize pathogens", ("Directly dissolving toxins forever", "Replacing red blood cells", "Blocking all body temperature changes"), "Vaccines stimulate immune memory against disease-causing agents.", "NCERT Science / public health"),
            FactItem("Science", "Quantum computing uses what basic unit of information?", "Qubit", ("Gigabit", "Neuron", "Photon cell"), "Unlike classical bits, qubits can exist in superposition.", "Science / emerging tech"),
            FactItem("Science", "Which gas is most abundant in the Earth's atmosphere?", "Nitrogen", ("Oxygen", "Carbon dioxide", "Argon"), "Nitrogen makes up roughly 78 percent of the atmosphere.", "NCERT Science / general science"),
        ],
        "Current Affairs": [
            FactItem("Current Affairs", "The Digital Personal Data Protection Act, 2023 deals primarily with what?", "Protection and processing of digital personal data", ("Regulation of mining leases", "Electoral delimitation", "Agricultural tenancy"), "The DPDP Act establishes a framework for handling digital personal data.", "Current Affairs / governance-tech"),
            FactItem("Current Affairs", "The IndiaAI Mission approved in 2024 focuses on what?", "Compute capacity, datasets and indigenous AI innovation", ("Privatizing all public universities", "Replacing RBI with an AI regulator", "Creating a new Election Commission"), "The IndiaAI Mission supports the country's AI ecosystem.", "Current Affairs / science-tech policy"),
            FactItem("Current Affairs", "The new criminal laws replacing IPC, CrPC and Evidence Act include which law as replacement for IPC?", "Bharatiya Nyaya Sanhita", ("Bharatiya Vyapar Sanhita", "Nagarik Nyaya Adhiniyam", "Bharatiya Samvidhan Adhiniyam"), "The Bharatiya Nyaya Sanhita replaces the Indian Penal Code.", "Current Affairs / legal reforms"),
            FactItem("Current Affairs", "The Global Biofuel Alliance was launched during which forum under India's leadership?", "G20", ("SAARC", "SCO", "WTO"), "The alliance was launched during India's G20 presidency.", "Current Affairs / summits"),
            FactItem("Current Affairs", "The New Delhi Leaders' Declaration is associated with which summit?", "G20 Summit under India's presidency in 2023", ("BRICS Summit 2025", "COP26", "WTO Ministerial 2022"), "The declaration was adopted at the G20 Leaders' Summit in New Delhi in 2023.", "Current Affairs / international relations"),
            FactItem("Current Affairs", "Which countries make up the Quad?", "India, Japan, Australia and the United States", ("India, Russia, Australia and Japan", "India, France, Australia and the United States", "India, Indonesia, Japan and the United States"), "The Quad is an Indo-Pacific grouping of India, Japan, Australia and the United States.", "Current Affairs / strategic groupings"),
            FactItem("Current Affairs", "Blue Economy refers to what?", "Sustainable use of ocean resources for growth and ecosystem health", ("Only offshore oil drilling", "Full privatization of ports", "Military control of the Indian Ocean"), "The blue economy links livelihoods and growth with marine sustainability.", "Current Affairs / economy-environment"),
            FactItem("Current Affairs", "National Green Hydrogen Mission is linked to which broad goal?", "Clean energy transition and industrial decarbonisation", ("Expansion of coal subsidy", "Urban rent control", "Privatization of forests"), "The mission seeks to build a green hydrogen ecosystem.", "Current Affairs / climate and industry"),
        ],
    }


FACT_BANK = _facts()


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain JSON.")
    return text[start : end + 1]


@lru_cache(maxsize=1)
def _load_research_notes() -> str:
    return NOTES_PATH.read_text(encoding="utf-8")


def _rag_paths(corpus: str) -> tuple[Path, Path]:
    if corpus == "current_affairs":
        return CURRENT_RAG_VECTOR_INDEX_PATH, CURRENT_RAG_INDEX_PATH
    return RAG_VECTOR_INDEX_PATH, RAG_INDEX_PATH


def _select_rag_corpus(subject: str) -> str:
    if subject == "Current Affairs":
        return "current_affairs"
    return "static"


@lru_cache(maxsize=2)
def _load_rag_index(corpus: str = "static") -> dict[str, Any] | None:
    vector_path, index_path = _rag_paths(corpus)
    path = vector_path if vector_path.exists() else index_path
    if not path.exists():
        return None
    index = load_index(path)
    if index.get("kind") == "redirect":
        target = path.parent / index["target"]
        if target.exists():
            index = load_index(target)
        else:
            return None
    if index.get("kind") != "vector":
        return None
    return index


def _extract_markdown_section(notes: str, heading: str, level: int = 2) -> str:
    prefix = "#" * level
    pattern = rf"(?ms)^{re.escape(prefix)} {re.escape(heading)}\n(.*?)(?=^#{{1,{level}}} |\Z)"
    match = re.search(pattern, notes)
    return match.group(1).strip() if match else ""


def _subject_book_hint(subject: str) -> str:
    hints = {
        "History": "Use NCERT History, Spectrum, and Nitin Singhania style factual grounding.",
        "Polity": "Use Laxmikanth style constitutional and institutional grounding.",
        "Economy": "Use NCERT Economics style basics plus recent Indian policy and reform themes.",
        "Geography": "Use NCERT Geography and G.C. Leong style physical and Indian geography grounding.",
        "Environment": "Use Shankar IAS and NCERT style ecology, conservation and convention grounding.",
        "Science": "Use NCERT science basics plus India-focused science-tech and space developments.",
        "Current Affairs": "Use the last decade topic map and keep the question tightly tied to recent Indian developments.",
    }
    return hints[subject]


def _extract_subject_context(subject: str) -> str:
    notes = _load_research_notes()
    standard_sources = _extract_markdown_section(notes, "Standard GS book backbone used")
    current_affairs = _extract_markdown_section(notes, "Last 10 years current-affairs topic map")
    quiz_design = _extract_markdown_section(notes, "How the quiz was designed")

    relevant_lines: list[str] = []
    keywords = SUBJECT_KEYWORDS[subject]
    for line in current_affairs.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            relevant_lines.append(line.strip())

    relevant_current = "\n".join(relevant_lines[:12]).strip()
    parts = [
        _subject_book_hint(subject),
        "Source backbone excerpt:",
        standard_sources,
        "Relevant current-affairs/topic excerpt:",
        relevant_current or "Use a static-concept-led question if no strong current-affairs match appears.",
        "UPSC pattern excerpt:",
        quiz_design,
        "Keep the question factual, conservative, and suitable for UPSC Prelims General Studies.",
    ]
    return "\n".join(part for part in parts if part).strip()


def _embed_query(text: str) -> list[float]:
    index = _load_rag_index("current_affairs") or _load_rag_index("static")
    embed_model = index.get("embedding_model", OLLAMA_EMBED_MODEL) if index else OLLAMA_EMBED_MODEL
    if isinstance(embed_model, str) and embed_model.startswith("hashed-bow-"):
        return hash_text_to_vector(text)
    payload = {"model": embed_model, "input": [text]}
    req = Request(
        f"{OLLAMA_BASE_URL}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
        raise ValueError("Embedding query failed.")
    return embeddings[0]


def _build_rag_query(subject: str) -> str:
    query_map = {
        "History": "NCERT history modern india culture themes in history UPSC prelims",
        "Polity": "NCERT political science constitution parliament rights federalism UPSC prelims",
        "Economy": "NCERT indian economic development reforms planning inflation UPSC prelims",
        "Geography": "NCERT geography india physical environment rivers climate monsoon UPSC prelims",
        "Environment": "NCERT geography environment biodiversity conservation climate UPSC prelims",
        "Science": "NCERT science space technology basic science UPSC prelims",
        "Current Affairs": "India recent policy missions laws summits space climate digital governance official updates UPSC prelims",
    }
    return query_map[subject]


def _prefer_official_current_affairs_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(chunk: dict[str, Any]) -> tuple[int, float]:
        url = str(chunk.get("url") or "").lower()
        path = chunk.get("path")
        title = str(chunk.get("title") or "").lower()
        is_local_note = bool(path) or "research notes" in title
        is_official = any(domain in url for domain in ("pib.gov.in", "isro.gov.in", "g20.org", "upsc.gov.in"))
        priority = 0 if is_official else 1 if not is_local_note else 2
        return (priority, -float(chunk.get("score") or 0.0))

    return sorted(chunks, key=rank_key)


def _extract_rag_context(subject: str) -> tuple[str, str | None, str | None]:
    corpus = _select_rag_corpus(subject)
    index = _load_rag_index(corpus)
    if not index:
        return "", None, None

    query_embedding = _embed_query(_build_rag_query(subject))
    chunks = search_vector_index(index, query_embedding, subject=subject, top_k=6)
    if not chunks:
        return "", None, None
    if corpus == "current_affairs":
        chunks = _prefer_official_current_affairs_chunks(chunks)

    heading = "Retrieved current-affairs context:" if corpus == "current_affairs" else "Retrieved NCERT context:"
    parts = [heading]
    primary_source = chunks[0]["title"]
    primary_text = chunks[0]["text"]
    primary_chunk = chunks[0]
    parts.append(f'Source: {primary_chunk["title"]}')
    parts.append(primary_chunk["text"][:700].strip())
    return "\n".join(parts).strip(), primary_source, primary_text


def _extract_labeled_field(text: str, label: str) -> str:
    pattern = rf"(?im)^{re.escape(label)}\s*:\s*(.+)$"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing field: {label}")
    return match.group(1).strip()


def _build_explanation_from_context(subject: str, primary_source: str | None, primary_text: str | None) -> str:
    if not primary_text:
        return f"Grounded in {subject} context prepared for UPSC-style practice."
    sentence = re.split(r"(?<=[.!?])\s+", primary_text.strip())[0]
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) > 240:
        sentence = sentence[:237].rstrip() + "..."
    if primary_source:
        return f"{sentence} Source used: {primary_source}."
    return sentence


def _normalize_structured_question(
    raw_text: str,
    subject: str,
    primary_source: str | None,
    primary_text: str | None,
) -> dict[str, Any]:
    question = _extract_labeled_field(raw_text, "QUESTION")
    correct = _extract_labeled_field(raw_text, "CORRECT")
    wrong1 = _extract_labeled_field(raw_text, "WRONG1")
    wrong2 = _extract_labeled_field(raw_text, "WRONG2")
    wrong3 = _extract_labeled_field(raw_text, "WRONG3")
    explanation = _build_explanation_from_context(subject, primary_source, primary_text)
    reference = primary_source or f"{subject} / retrieved context"

    options = [correct, wrong1, wrong2, wrong3]
    if len({option.strip().lower() for option in options}) != 4:
        raise ValueError("Options were not distinct.")

    rng = random.Random()
    rng.shuffle(options)
    answer_index = options.index(correct)

    return {
        "question": question,
        "options": options,
        "answer_index": answer_index,
        "explanation": explanation,
        "subject": subject,
        "reference": reference,
    }


def _validate_question(item: dict[str, Any]) -> dict[str, Any]:
    required = {"question", "options", "answer_index", "explanation", "subject", "reference"}
    if not required.issubset(item):
        raise ValueError("Question object missing required fields.")
    options = item["options"]
    if not isinstance(options, list) or len(options) != 4 or len(set(options)) != 4:
        raise ValueError("Each question must have exactly 4 distinct options.")
    answer_index = item["answer_index"]
    if not isinstance(answer_index, int) or answer_index not in range(4):
        raise ValueError("answer_index must be 0..3.")
    return {
        "question": str(item["question"]).strip(),
        "options": [str(option).strip() for option in options],
        "answer_index": answer_index,
        "explanation": str(item["explanation"]).strip(),
        "subject": str(item["subject"]).strip(),
        "reference": str(item["reference"]).strip(),
    }


def _call_ollama_raw_question(subject: str) -> str:
    rag_context, primary_source, primary_text = _extract_rag_context(subject)
    subject_context = rag_context or _extract_subject_context(subject)
    prompt = f"""
Create exactly one UPSC-style multiple choice question for UPSC General Studies.

Requirements:
- The subject must be exactly "{subject}".
- Return plain text only in exactly this 5-line format:
QUESTION: ...
CORRECT: ...
WRONG1: ...
WRONG2: ...
WRONG3: ...
- Keep it objective, UPSC-like, and factually grounded.
- Prefer statement-style or elimination-friendly framing when natural.
- Keep all 4 options short and distinct.
- Use the retrieved context below as the main factual grounding when available.
- Use only facts that are supported by the retrieved context below.
- Do not add numbering, bullets, markdown, or any extra text.

Use this subject-specific grounding context:
{subject_context}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 140,
        },
    }
    req = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "").strip(), primary_source, primary_text


def _call_ollama_question() -> dict[str, Any]:
    last_error: Exception | None = None
    subject = random.choice(SUBJECTS)
    for _attempt in range(QUESTION_RETRY_LIMIT):
        try:
            raw_text, primary_source, primary_text = _call_ollama_raw_question(subject)
            item = _normalize_structured_question(raw_text, subject, primary_source, primary_text)
            return _validate_question(item)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError("Model did not return a valid question.") from last_error


def generate_quiz_payload(force_fallback: bool = False) -> dict[str, Any]:
    if force_fallback:
        raise ValueError("Fallback mode is disabled. Ollama generation is required.")

    try:
        question = _call_ollama_question()
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to generate question with Ollama {OLLAMA_MODEL}. "
            "Please ensure Ollama is running and the model can complete the request."
        ) from exc

    return {
        "source": "ollama + vector-rag",
        "model": OLLAMA_MODEL,
        "count": 1,
        "question": question,
    }
