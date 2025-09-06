# !pip install sentence-transformers rank-bm25 rapidfuzz pdfminer.six scikit-learn huggingface_hub[hf_xet] numpy

import json, re, math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

RUS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
ENG = "abcdefghijklmnopqrstuvwxyz"
DIG = "0123456789"


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(rf"[^ {RUS}{ENG}{DIG}\-+#_/\.]", " ", s)
    return s.strip()


SKILL_CANON = {
    # языки/стэк
    "python", "r", "java", "kotlin", "javascript", "typescript", "go", "rust", "c", "c++", "c#", "sql", "bash", "matlab", "lua", "swift", "perl",
    # фронт
    "react", "redux", "redux-saga", "react query", "next.js", "vite", "webpack", "zustand", "mobx", "antd", "material ui", "tailwind css", "chart.js", "d3", "vue.js", "angular", "backbone.js", "jquery", "html5", "css3", "sass", "less", "bootstrap",
    # мобильная
    "android", "ios", "swift", "react native", "flutter", "xamarin", "ionic", "phonegap", "cordova", "objective-c", "kotlin", "material design", "mobile development",
    # бек
    "node.js", "django", "flask", "fastapi", "spring", "grpc", "rest", "graphql", "express.js", "ruby on rails", "laravel", "asp.net", "apache", "tomcat", "jetty", "play framework", "hibernate",
    # devops
    "docker", "docker-compose", "kubernetes", "nginx", "gitlab ci", "github actions", "ci/cd", "terraform", "ansible", "puppet", "chef", "vagrant", "prometheus", "grafana", "elk stack", "azure devops", "jenkins", "vault", "helm", "cloudformation", "cloudwatch", "k8s",
    # data
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "xgboost", "airflow", "spark", "hadoop", "apache kafka", "mongodb", "mysql", "postgresql", "elasticsearch", "redis", "sqldb", "bigquery", "tableau", "power bi", "matplotlib", "seaborn", "plotly", "bokeh", "databricks",
    # bio
    "геномика", "биоинформатика", "rna-seq", "dna-seq", "single-cell", "aligners", "gatk", "samtools", "plink", "metagenomics", "CRISPR", "bioinformatics tools", "bioinformatics pipelines", "next-generation sequencing", "proteomics", "epigenetics",
    # аналитика/общие
    "аналитика данных", "etl", "sql", "bi", "tableau", "power bi", "data mining", "data wrangling", "data engineering", "data visualization", "machine learning", "deep learning", "predictive analytics", "statistics", "A/B testing", "business intelligence", "customer analytics", "market research", "reporting",
    # искусственный интеллект
    "computer vision", "natural language processing", "reinforcement learning", "neural networks", "generative adversarial networks", "opencv", "spaCy", "transformers", "nltk", "bert", "gpt-3", "vgg", "resnet", "yolo", "deep reinforcement learning", "ai ethics", "tensorflow lite", "pytorch lightning",
    # cloud
    "aws", "azure", "google cloud", "cloud computing", "cloud storage", "cloud security", "lambda", "s3", "ec2", "gcp", "firebase", "cloud firestore", "cloud run", "kubernetes engine", "cloud functions", "cloud sql", "cloud storage",
    # блокчейн
    "ethereum", "smart contracts", "solidity", "blockchain", "cryptocurrency", "web3", "ipfs", "hyperledger", "distributed ledger", "bitcoin", "cryptography", "ethereum dapps", "decentralized finance", "nft", "blockchain development", "dao", "cryptographic algorithms",
    # другие
    "scrum", "agile", "kanban", "devops culture", "software architecture", "project management", "jira", "confluence", "git", "github", "gitlab", "bitbucket", "terraform", "aws cloudformation", "apache spark", "elasticsearch", "logstash", "kafka", "hibernate", "graphql", "swagger", "graph databases",
    # дизайн
    "ui/ux design", "ux research", "design thinking", "user interface", "user experience", "sketch", "figma", "adobe xd", "illustrator", "photoshop", "autocad", "3ds max", "blender", "maya", "revit", "solidworks", "cinema 4d",
}


SKILL_CANON_NORM = sorted({normalize(x) for x in SKILL_CANON})


def extract_skills(text: str, extra_tokens=None):
    text_n = normalize(text)
    tokens = set(re.split(r"[,\s/|]+", text_n))
    raw = set()
    for t in tokens:
        if not t or len(t) < 2: 
            continue
        if t in SKILL_CANON_NORM:
            raw.add(t)
    candidates = set()
    toks = [t for t in tokens if t]
    phrases = set()
    for n in (2,3):
        for i in range(max(0,len(toks)-n+1)):
            phrases.add(" ".join(toks[i:i+n]))
    phrases |= tokens

    for p in phrases:
        match, score, _ = process.extractOne(p, SKILL_CANON_NORM, scorer=fuzz.token_set_ratio)
        if score >= 88:
            candidates.add(match)
    return sorted(raw | candidates)


def job_doc(v):
    parts = []
    push = parts.append

    # title (высокий вес)
    push(("title: " + v.get("post","")) * 3)
    # tags (высокий вес)
    push(("tags: " + " ".join(v.get("tags",[]))) * 3)
    # responsibilities / requirements (средний)
    req = " ".join(v.get("requirements",{}).get("description",[]))
    resp = " ".join(v.get("responsibilities",{}).get("description",[]))
    push(("requirements: " + req) * 2)
    push(("responsibilities: " + resp) * 2)
    # метаданные
    push("industry: " + v.get("company",{}).get("industry",""))
    push("region: " + v.get("region",""))
    push("company: " + v.get("company",{}).get("name",""))
    push("salary: " + v.get("salary",""))
    return normalize(" \n".join(parts))


EMB_MODEL_NAME = "intfloat/multilingual-e5-base"
embedder = SentenceTransformer(EMB_MODEL_NAME, device="cpu")

def emb_query(text: str):
    return embedder.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)


def emb_passage(text: str):
    return embedder.encode(f"passage: {text}", normalize_embeddings=True, show_progress_bar=False)


def jaccard(a:set,b:set):
    if not a and not b: return 0.0
    return len(a&b)/max(1,len(a|b))


LEVEL_MAP = {
    "junior": 0, "младший": 0, "стажер": 0, "интерн": 0,
    "middle": 1, "мидл": 1, "средний": 1,
    "senior": 2, "сеньор": 2, "ведущий": 2, "lead": 2, "тимлид": 2,
}

def infer_level(text: str):
    t = normalize(text)
    best = None
    for k, val in LEVEL_MAP.items():
        if re.search(rf"\b{k}\b", t):
            best = max(best, val) if best is not None else val
    return best


def level_score(resume_text, job_text):
    r = infer_level(resume_text)
    j = infer_level(job_text)
    if j is None or r is None:
        return 0.5
    return 1.0 if r==j else (0.7 if abs(r-j)==1 else 0.3)


def region_score(resume_text, job_region):
    t = normalize(resume_text)
    willing = any(w in t for w in ["готов переезду","готов к переезду","готов командировкам","готов к командировкам","relocate","переезд"])
    if not job_region: 
        return 0.5
    if job_region and willing:
        return 0.8
    if job_region.lower() in t:
        return 1.0
    return 0.5


vacancies = json.load(open('C:/Users/honor/Downloads/Telegram Desktop/vacancies.json', 'r', encoding='utf-8'))

job_texts = [job_doc(v) for v in vacancies]
bm25 = BM25Okapi([jt.split() for jt in job_texts])


resume_pdf_path = 'C:/Users/honor/Downloads/Telegram Desktop/Резюме_Frontend_разработчик_Егор_Яровицын_от_25_06_2025_09_51.pdf'
resume_text_raw = extract_text(resume_pdf_path) or ""
resume_text = normalize(resume_text_raw)

resume_emb = emb_query(resume_text)
resume_skills = set(extract_skills(resume_text))

job_embs = [emb_passage(t) for t in job_texts]
job_skills = []
job_titles = []
for v in vacancies:
    title = normalize(v.get("post",""))
    job_titles.append(title)
    full = job_doc(v)
    job_skills.append(set(extract_skills(" ".join([
        v.get("post",""),
        " ".join(v.get("tags",[])),
        " ".join(v.get("requirements",{}).get("description",[])),
        " ".join(v.get("responsibilities",{}).get("description",[])),
    ]))))

# ROLE_TERMS = ["frontend","front-end","фронтенд","backend","бекенд","full-stack","фулстек","биоинформатика","data","ml","ios","android","devops"]
ROLE_TERMS = [
    # Разработка фронтенда
    "frontend", "front-end", "фронтенд", "react", "vue.js", "angular", "html", "css", "javascript", "typescript", "redux", "webpack", "next.js", "vite", "material ui", "tailwind css", "sass", "less", "bootstrap", "ui/ux design", "responsive design", "css3", "html5", 
    # Разработка бэкенда
    "backend", "бекенд", "node.js", "django", "flask", "spring", "java", "python", "ruby", "c#", "graphql", "rest api", "microservices", "express.js", "php", "java spring", "ruby on rails", "flask", "fastapi", "nestjs", "go", "scala", 
    # Full-stack разработка
    "full-stack", "фулстек", "web development", "server-side", "client-side", "node.js", "react", "angular", "vue.js", "python", "java", "javascript", "mongodb", "postgresql", "mysql", "docker", "kubernetes", "graphql", "ci/cd", "devops", 
    # Мобильная разработка
    "ios", "android", "react native", "swift", "kotlin", "flutter", "xamarin", "ionic", "mobile development", "mobile apps", "app development", "objective-c", 
    # Машинное обучение и аналитика данных
    "ml", "machine learning", "data science", "data analysis", "data engineer", "data scientist", "deep learning", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "xgboost", "spark", "etl", "etl pipelines", "data wrangling", "data visualization", 
    # Искусственный интеллект
    "ai", "artificial intelligence", "neural networks", "computer vision", "natural language processing", "reinforcement learning", "deep learning", "transformers", "bert", "gpt-3", "opencv", "spaCy", "nltk", "text mining", "image recognition", "speech recognition", "chatbots", 
    # DevOps и облачные технологии
    "devops", "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd", "nginx", "gitlab", "jenkins", "cloud", "cloud architecture", "cloud computing", "cloud infrastructure", "nginx", "cloud storage", "cloud security", "cloud migration", "cloud deployment", 
    # Блокчейн
    "blockchain", "ethereum", "smart contracts", "solidity", "web3", "cryptocurrency", "nft", "ipfs", "decentralized applications", "hyperledger", "distributed ledger", "blockchain development", "bitcoin", "blockchain security", "blockchain architecture", 
    # Аналитика и бизнес-интеллект
    "bi", "business intelligence", "data analytics", "tableau", "power bi", "sas", "d3.js", "qlik", "business analyst", "etl processes", "dashboard", "reporting", "analytics", "data visualization", 
    # Биоинформатика
    "биоинформатика", "genomics", "rna-seq", "dna-seq", "next-generation sequencing", "bioinformatics tools", "gatk", "samtools", "biostatistics", "biological data analysis", "clinical data analysis", 
    # Прочее
    "ux design", "ui design", "web design", "graphic design", "3d modeling", "cad", "photoshop", "illustrator", "blender", "maya", "autocad", "sketch", "figma", "adobe xd", "solidworks", "cgi", "3ds max"
]
resume_role = " ".join([t for t in ROLE_TERMS if t in resume_text])

vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
X = vec.fit_transform([resume_role] + job_titles)
tfidf_resume = X[0]
tfidf_jobs = X[1:]
title_sims = (tfidf_jobs @ tfidf_resume.T).toarray().ravel()
if title_sims.size == 0:
    title_sims = np.zeros(len(vacancies))

bm25_scores = bm25.get_scores(resume_text.split())
bm_scaler = MinMaxScaler()
bm_norm = bm_scaler.fit_transform(bm25_scores.reshape(-1,1)).ravel()


def cos(u,v): 
    return float(np.dot(u,v))


emb_sims = np.array([cos(resume_emb, e) for e in job_embs])
emb_norm = (emb_sims - emb_sims.min()) / (max(1e-9, emb_sims.max()-emb_sims.min()))

skill_sims = np.array([jaccard(resume_skills, s) for s in job_skills])

level_sims = np.array([
    level_score(resume_text, job_texts[i]) for i in range(len(vacancies))
])

region_sims = np.array([
    region_score(resume_text, vacancies[i].get("region","")) for i in range(len(vacancies))
])

W_EMB   = 0.50
W_SKILL = 0.25
W_BM25  = 0.10
W_TITLE = 0.08
W_LEVEL = 0.04
W_REGION= 0.03

final_score = (
    W_EMB   * emb_norm +
    W_SKILL * skill_sims +
    W_BM25  * bm_norm +
    W_TITLE * title_sims +
    W_LEVEL * level_sims +
    W_REGION* region_sims
)

order = np.argsort(-final_score)


def pretty_pct(x): 
    return f"{100*x:.1f}%"


filtered_ids = []
for rank, idx in enumerate(order[:5], 1):
    if final_score[idx] > 0.55:
        v = vacancies[idx]
        id = v['id']
        filtered_ids.append(id)
        print(f"{rank}. {v['company']['name']} — {v['post']} [{v.get('region','')}]  | score={final_score[idx]:.3f}")
        print(f"   emb={pretty_pct(emb_norm[idx])} | skills={pretty_pct(skill_sims[idx])} | bm25={pretty_pct(bm_norm[idx])} | title={pretty_pct(title_sims[idx])} | level={pretty_pct(level_sims[idx])} | region={pretty_pct(region_sims[idx])}")
        inter = sorted(list(resume_skills & job_skills[idx]))
        if inter:
            print(f"   matched skills: {', '.join(inter[:15])}{'…' if len(inter)>15 else ''}")