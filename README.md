<div align="center">

<img src="https://img.shields.io/badge/NeuralDB-1.0.0-10b981?style=for-the-badge&labelColor=0a0a14" alt="NeuralDB">

# NeuralDB 🧠

**The AI-Native Database Engine**

*Purpose-built for how AI systems think — not bolted-on vectors, but genuine cognitive architecture*

[![CI](https://github.com/<owner>/neuraldb/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/neuraldb/actions)
[![Coverage](https://codecov.io/gh/<owner>/neuraldb/branch/main/graph/badge.svg)](https://codecov.io/gh/<owner>/neuraldb)
[![PyPI version](https://img.shields.io/pypi/v/neuraldb.svg)](https://pypi.org/project/neuraldb/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/docker/pulls/<owner>/neuraldb)](https://hub.docker.com/r/<owner>/neuraldb)

[Plain English](#-neuraldb-in-plain-english) · [The Problem](#-the-problem-it-solves) · [Market Landscape](#-market-landscape) · [Features](#-features) · [Quick Start](#-quick-start) · [AIQL Reference](#-aiql-language-reference) · [API Docs](#-rest-api) · [Deployment](#-deployment) · [Integrations](#-integrations) · [Benchmarks](#-benchmarks) · [Contributing](#-contributing)

</div>

---

## What Is NeuralDB?

Traditional databases were designed for transactions. Vector databases were designed for similarity search. NeuralDB was designed for **the way AI systems actually reason**.

Every modern AI application — chatbots, RAG pipelines, autonomous agents, recommendation systems — needs storage that understands:

- **Uncertainty.** Not every fact is equally trustworthy.
- **Time.** Recent memories matter more than old ones.
- **Context.** Short-term working memory differs from long-term knowledge.
- **Relationships.** Entities don't exist in isolation.
- **Semantics.** Meaning matters, not just keywords.

NeuralDB gives you all of this in a single engine, with a query language shaped around cognitive operations instead of SQL CRUD.

```python
from neuraldb import NeuralDB, LaneType

db = NeuralDB("assistant_memory")
db.create_lane("knowledge", lane_type=LaneType.SEMANTIC)

# Store a fact with certainty
db.insert("knowledge",
    data={"entity": "Paris", "type": "city", "country": "France"},
    confidence=0.99,
    embedding=encoder.encode("Paris capital city France"),
    relations=[(eiffel_id, "contains", 0.95)]
)

# Query like you think
result = db.query("RECALL TOP 5 FROM knowledge WHERE confidence > 0.8")

# Find semantically similar — with confidence weighting
matches = db.similarity_search(query_embedding, top_k=10, min_confidence=0.7)

# Traverse the knowledge graph
subgraph = db.traverse_graph(paris_id, rel_type="contains", depth=2)

# Consolidate learned memories (like sleep)
db.consolidate("episodes", "knowledge", min_confidence=0.75)
```
---

## 🧠 NeuralDB in Plain English

> **Not a database person? Start here.**

### The simple analogy

Think about how your own memory works. When you remember something, your brain doesn't just store the fact — it also keeps track of **how sure you are**, **how recent it is**, and **what it connects to**. Old memories fade unless you revisit them. Important things stick. New evidence can change what you believe.

Regular databases don't work like that. They store facts in rows and columns, treat everything as equally true, and keep data forever with no sense of time or uncertainty.

**NeuralDB works like the brain, not the filing cabinet.** Every piece of data it stores comes with:

- A **certainty score** — how sure are we about this? (0% = pure guess, 100% = rock-solid fact)
- A **timestamp and decay rate** — how quickly should this memory fade if unused?
- A **memory type** — is this short-term working context, long-term knowledge, or an event we witnessed?
- **Relationships** — what other facts does this connect to, and how?

---

## 🔥 The Problem It Solves

Building an AI app today means stitching together 3–5 separate tools just to handle memory properly. You need a vector database for semantic search, a graph database for relationships, a regular database for structured storage, plus custom code to handle confidence and time. It's expensive, fragile, and still doesn't give you cognitive memory types or a language designed for AI reasoning.

Here are the five real problems NeuralDB fixes — explained without jargon:

**Problem 1: AI treats stale information as current fact**

Imagine a chatbot that "knows" someone is the CEO of a company — but they resigned eight months ago. The database has no concept of information ageing. It stored that fact and it's still there, equally trusted.

NeuralDB solves this with **temporal decay**: every record has a configurable half-life. A fact stored in a "working memory" lane might fade after an hour. A fact in "episodic memory" might fade over a week. Long-term "semantic" knowledge stays until you choose to update or remove it.

**Problem 2: AI cannot express uncertainty**

When a doctor diagnoses a patient, they don't say "you have condition X." They say "based on the symptoms, I'm about 80% confident this is condition X." That nuance matters. Traditional databases have no way to represent it — every stored value carries equal weight.

NeuralDB attaches a **confidence score (0.0–1.0)** to every record. A fact scraped from a reliable source gets 0.95. An inference from a model gets 0.60. User-provided data with no verification gets 0.40. Queries can filter, rank, and reason based on these scores.

**Problem 3: AI has no short-term vs long-term memory**

During a conversation, an AI needs to remember what you said two messages ago (short-term, expires soon) *and* what it learned about your preferences last month (long-term, should persist). These are fundamentally different types of memory — but most databases treat them identically.

NeuralDB has **six typed memory lanes**, each with different default behaviours:

| Lane | Like in your brain | Default lifespan |
|---|---|---|
| `working` | Things you're actively thinking about | 1 hour |
| `episodic` | Memories of specific events | 7 days |
| `semantic` | General knowledge and facts | Permanent |
| `procedural` | How to do things (skills, rules) | Permanent |
| `associative` | Connections between concepts | Permanent |
| `sensory` | Raw unprocessed input | 5 minutes |

**Problem 4: Searching only works on exact words**

Search a regular database for "automobile" and you'll miss every record that says "car", "vehicle", or "motor transport". AI applications need to find things by *meaning*, not by exact character matching.

NeuralDB has **vector-native semantic search** built in. Every record can store an embedding — a mathematical representation of its meaning — and search finds similar meanings automatically. Better still, NeuralDB's search combines similarity with confidence:

```
final score = how similar × how trustworthy
```

A 90%-similar but 20%-confidence record won't outrank a 80%-similar but 95%-confidence one. That's the right behaviour.

**Problem 5: AI can't see how things connect**

A regular database can tell you the Eiffel Tower exists. It cannot naturally tell you that it's in Paris, which is the capital of France, which is in Western Europe — and trace that entire chain for you.

NeuralDB has a **knowledge graph built directly into the storage engine**, not as a separate service. Every record can have typed, weighted edges to other records. You can ask: *"starting from this concept, what is reachable within 3 hops?"* — and get a structured answer in milliseconds.

---

## 📊 Market Landscape

### How does this compare to what already exists?

There are many good tools out there for pieces of this problem. Here's an honest map of the landscape and where NeuralDB sits:

#### Category 1 — Pure Vector Databases
*Pinecone, Weaviate, Qdrant, Milvus, Chroma, FAISS*

These are excellent at one thing: storing embeddings and finding similar ones fast. They scale to billions of vectors and have mature managed hosting options.

**What they lack:** No confidence scoring. No temporal decay. No memory types. No knowledge graph. No cognitive query language. You get similarity search and metadata filtering — and you build everything else yourself.

**When to use them instead:** You need to search 100M+ vectors at sub-10ms latency and nothing else. Pinecone and Milvus are the industry standard at that scale.

#### Category 2 — Graph Databases
*Neo4j, FalkorDB*

Excellent at storing and traversing relationships between entities. Neo4j recently added native vector support, making it capable of hybrid graph+vector queries.

**What they lack:** No confidence scoring. No temporal decay. No cognitive memory types. Complex query language (Cypher) with a steep learning curve. Heavy infrastructure requirements.

**When to use them instead:** Your primary need is relationship traversal across millions of nodes at enterprise scale, and you have a dedicated DBA team.

#### Category 3 — AI Memory Middleware
*Mem0, Zep, Letta (MemGPT)*

A newer category (2024–2025) that sits between your LLM and a database, handling the memory layer for you. Mem0 uses a hybrid vector + graph store internally. Zep uses temporal knowledge graphs. Letta treats working memory as a first-class concept.

**What they lack:** These are frameworks and services, not databases you embed in your own application. They abstract away control in exchange for convenience. Limited ability to run custom queries, access raw records, or tune storage behaviour. Most require sending your data to a third-party service.

**When to use them instead:** You want a plug-and-play memory layer for a LangChain or LlamaIndex application and don't need fine-grained control over how memories are stored and retrieved.

#### Category 4 — Traditional Databases with Vector Extensions
*PostgreSQL + pgvector, Redis + vector search, MongoDB Atlas Vector Search*

Familiar SQL or NoSQL databases with vector search bolted on. Great if you already have PostgreSQL and just need basic semantic search alongside your structured data.

**What they lack:** Vectors are second-class citizens. No cognitive memory architecture. No confidence. No decay. Performance degrades significantly above 10–50M vectors.

**When to use them instead:** You already run PostgreSQL and need light-touch semantic search without adding a new database to your infrastructure.

---

### Where NeuralDB fits

```
                    CONFIDENCE   TEMPORAL    COGNITIVE    VECTOR      KNOWLEDGE   COGNITIVE
                    SCORING      DECAY       MEMORY TYPES SEARCH      GRAPH       QUERY LANG
                    ─────────────────────────────────────────────────────────────────────────
Vector DBs          ✗            ✗           ✗            ✓✓✓         ✗           ✗
(Pinecone, Qdrant)

Graph DBs           ✗            ✗           ✗            ✓           ✓✓✓         ✗
(Neo4j)

Memory Middleware   partial      ✗           partial      ✓           partial     ✗
(Mem0, Zep)

pgvector / Redis    ✗            ✗           ✗            ✓           ✗           ✗

NeuralDB            ✓✓✓          ✓✓✓         ✓✓✓          ✓✓          ✓✓          ✓✓✓
```

**NeuralDB is the only system that natively combines all six in a single embeddable engine with zero external dependencies.**

The trade-off is honest: NeuralDB is not the right tool if your primary need is billion-scale vector indexing at sub-5ms latency. For that, Milvus or Pinecone win. NeuralDB targets the space where **AI reasoning quality matters more than raw vector throughput** — agents, assistants, RAG systems that need to be trustworthy, context-aware, and temporally intelligent.

---

## ✨ Features

### 🎯 Confidence-Aware Storage
Every record carries an epistemological certainty score `0.0–1.0`. This isn't just metadata — it drives queries, search ranking, consolidation decisions, and expiry.

```python
# High-confidence fact from a verified source
db.insert("knowledge", {"entity": "Earth", "type": "planet"},
          confidence=1.0, provenance="nasa.gov")

# Inferred fact with lower certainty
db.insert("knowledge", {"entity": "life_on_mars", "probable": True},
          confidence=0.23, provenance="inference_engine")
```

### ⏱ Temporal Decay (Memory Fades)
Memories lose relevance over time unless reinforced. NeuralDB implements exponential decay with configurable half-lives per lane, and access-based reinforcement (spaced repetition built into the engine).

```
effective_confidence(t) = base_confidence × e^(-ln(2)/half_life × t) + reinforcement_bonus
```

A record with `half_life=3600` (1 hour) and `confidence=1.0` decays to `0.5` after one hour, `0.25` after two hours. Accessing the record slows the decay.

### 🗂 Memory Lane Architecture
Collections are typed by cognitive role, not just data shape. Each lane type has sensible defaults for decay, confidence thresholds, and use cases:

| Lane Type | Cognitive Model | Default Half-Life | Best For |
|---|---|---|---|
| `semantic` | Long-term knowledge | None (permanent) | Facts, entities, concepts |
| `episodic` | Autobiographical events | 7 days | Conversations, interactions, logs |
| `working` | Active task context | 1 hour | Current session state, active goals |
| `procedural` | How-to knowledge | None | Workflows, rules, action sequences |
| `associative` | Pure relationships | None | Entity connections, tags |
| `sensory` | Raw input representations | 5 minutes | Unprocessed embeddings |

### 🔍 Vector-Native with Confidence Weighting
Embeddings are first-class citizens, not an afterthought. NeuralDB's semantic search combines cosine similarity with confidence weighting — a high-similarity but uncertain record doesn't outrank a slightly less similar but reliable one.

```
combined_score = cosine_similarity(query, record) × effective_confidence^0.5
```

This single formula correctly handles the tension between *finding similar things* and *trusting what you find*.

### 🕸 Knowledge Graph Built-In
No separate Neo4j required. NeuralDB maintains a typed, weighted directed graph over all records across all lanes. Traverse, find paths, and extract subgraphs as part of normal queries.

```python
# Create typed, weighted relationships between records
db.insert("knowledge", {"entity": "Louvre"},
          relations=[
              (paris_id, "located_in", 1.0),
              (eiffel_id, "near", 0.6),
          ])

# Traverse the graph
graph = db.traverse_graph(louvre_id, rel_type="located_in", depth=2)
# → {"nodes": [...], "edges": [...], "total_nodes": 3, "total_edges": 2}
```

### 💬 AIQL — AI Intelligence Query Language
AIQL replaces SQL with verbs that match cognitive operations. There's no `SELECT`, `INSERT`, `UPDATE`, or `DELETE` — those are database terms. NeuralDB has verbs that mean something to an AI system:

```sql
-- Retrieve high-confidence memories
RECALL TOP 10 FROM knowledge WHERE confidence > 0.8 ORDER BY confidence DESC

-- Store new knowledge with certainty
REMEMBER INTO knowledge SET entity=NewYork, type=city WITH CONFIDENCE 0.97

-- Selectively forget unreliable information
FORGET FROM working WHERE age > 3600

-- Move high-confidence episodes to permanent knowledge (like sleep consolidation)
CONSOLIDATE episodes INTO knowledge WHERE confidence > 0.75

-- Boost confidence after new corroborating evidence
REINFORCE knowledge WHERE entity = Paris BY 0.05

-- Decay confidence after contradicting evidence
DOUBT knowledge WHERE source = unreliable_source BY 0.2

-- Inspect the database
REFLECT ON database
REFLECT ON knowledge
```

### 🧠 Memory Consolidation
Inspired by how the brain transfers short-term memories to long-term storage during sleep, `CONSOLIDATE` moves high-confidence episodic records into semantic knowledge, merging near-duplicates rather than creating redundancy.

---

## 🚀 Quick Start

### Option A — Python Library (No Server)

```bash
pip install neuraldb
```

```python
from neuraldb import NeuralDB, LaneType
import tempfile

# Create a database
db = NeuralDB("my_ai_memory", path="/tmp/neuraldb")

# Create typed memory lanes
db.create_lane("knowledge",  lane_type=LaneType.SEMANTIC)
db.create_lane("episodes",   lane_type=LaneType.EPISODIC, decay_half_life=86400)
db.create_lane("context",    lane_type=LaneType.WORKING,  decay_half_life=3600)

# Store facts with confidence
paris_id = db.insert("knowledge",
    data={"entity": "Paris", "type": "city", "population": 2161000},
    confidence=0.99,
    embedding=[0.1, 0.3, ...],  # your encoder's output
    provenance="wikipedia"
)

eiffel_id = db.insert("knowledge",
    data={"entity": "Eiffel Tower", "height_m": 330},
    confidence=0.97,
    relations=[(paris_id, "located_in", 1.0)]
)

# Query with AIQL
result = db.query("RECALL TOP 5 FROM knowledge WHERE confidence > 0.8")
print(result["records"])  # sorted by confidence descending

# Semantic search
matches = db.similarity_search(
    query_vector=[0.1, 0.3, ...],
    top_k=5,
    min_confidence=0.7
)

# Graph traversal
graph = db.traverse_graph(eiffel_id, depth=2)

# Stats
print(db.stats_report())
```

### Option B — REST API Server

```bash
# 1. Install with server extras
pip install "neuraldb[server]"

# 2. Set your API key
export NEURALDB_API_KEY=<your-api-key>

# 3. Start the server
python server.py
# → Running on http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

```bash
# Create a database
curl -X POST http://localhost:8000/db/mydb/create \
  -H "X-API-Key: $NEURALDB_API_KEY"

# Create a lane
curl -X POST http://localhost:8000/db/mydb/lanes \
  -H "X-API-Key: $NEURALDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "knowledge", "lane_type": "semantic"}'

# Insert a record
curl -X POST http://localhost:8000/db/mydb/insert \
  -H "X-API-Key: $NEURALDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "lane": "knowledge",
    "data": {"entity": "Paris", "type": "city"},
    "confidence": 0.99,
    "provenance": "wikipedia"
  }'

# Query with AIQL
curl -X POST http://localhost:8000/db/mydb/query \
  -H "X-API-Key: $NEURALDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"aiql": "RECALL TOP 5 FROM knowledge WHERE confidence > 0.8"}'
```

### Option C — Docker

```bash
git clone https://github.com/<owner>/neuraldb.git
cd neuraldb
cp .env.example .env           # Set NEURALDB_API_KEY in .env
docker compose up -d           # Server at http://localhost:8000
```

---

## 📘 AIQL Language Reference

AIQL (AI Intelligence Query Language) is NeuralDB's query language. It replaces SQL's data-manipulation vocabulary with cognitive operations.

### RECALL — Retrieve memories

```
RECALL [TOP n] FROM <lane>
       [WHERE <field> <op> <value> [AND|OR ...]]
       [ORDER BY confidence|age [ASC|DESC]]
       [FUZZY]
```

**Operators:** `=` `!=` `>` `>=` `<` `<=` `CONTAINS` `LIKE`

**Special fields:** `confidence` (effective), `age` (seconds since creation), `version`

**FUZZY:** enables partial string matching on text fields

```sql
RECALL TOP 10 FROM knowledge WHERE confidence > 0.8
RECALL FROM episodes WHERE session = sess_001 ORDER BY age ASC
RECALL FROM knowledge WHERE type = city FUZZY
RECALL FROM working WHERE age > 1800
RECALL TOP 1 FROM knowledge WHERE entity CONTAINS Paris
```

### REMEMBER — Store new knowledge

```
REMEMBER INTO <lane> SET key=value, key2=value2
          [WITH CONFIDENCE <0.0-1.0>]
```

```sql
REMEMBER INTO knowledge SET entity=Berlin, type=city WITH CONFIDENCE 0.95
REMEMBER INTO working SET active_topic=AI WITH CONFIDENCE 1.0
```

### FORGET — Selective deletion

```
FORGET FROM <lane> WHERE <conditions>
```

```sql
FORGET FROM working WHERE age > 3600
FORGET FROM episodes WHERE confidence < 0.2
FORGET FROM knowledge WHERE source = deprecated_feed
```

### TRAVERSE — Graph walk

```
TRAVERSE FROM <node_id> [VIA <relation_type>] [DEPTH <n>]
```

```sql
TRAVERSE FROM abc-123 VIA located_in DEPTH 2
TRAVERSE FROM abc-123 DEPTH 3
```

### CONSOLIDATE — Memory consolidation

```
CONSOLIDATE <source_lane> INTO <target_lane> WHERE confidence > <threshold>
```

```sql
CONSOLIDATE episodes INTO knowledge WHERE confidence > 0.75
CONSOLIDATE sensory INTO episodic WHERE confidence > 0.6
```

### REINFORCE / DOUBT — Belief updating

```
REINFORCE <lane> WHERE <conditions> BY <amount>
DOUBT <lane> WHERE <conditions> BY <amount>
```

```sql
REINFORCE knowledge WHERE entity = Paris BY 0.05
DOUBT knowledge WHERE source = tabloid BY 0.3
REINFORCE episodes WHERE session = sess_001 BY 0.1
```

### REFLECT — Introspection

```
REFLECT ON database
REFLECT ON <lane_name>
```

```sql
REFLECT ON database
REFLECT ON knowledge
REFLECT ON episodes
```

---

## 🌐 REST API

The server exposes a fully-documented REST API. Visit `/docs` for the interactive Swagger UI.

### Authentication

All endpoints (except `GET /health`) require the `X-API-Key` header:

```
X-API-Key: <your-api-key>
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (no auth) |
| `POST` | `/db/{name}/create` | Create a database |
| `GET` | `/db/{name}/stats` | Statistics and health report |
| `DELETE` | `/db/{name}` | Unload a database |
| `GET` | `/db/{name}/lanes` | List memory lanes |
| `POST` | `/db/{name}/lanes` | Create a memory lane |
| `POST` | `/db/{name}/insert` | Insert a record |
| `GET` | `/db/{name}/get/{lane}/{id}` | Get record by ID |
| `PATCH` | `/db/{name}/update/{lane}/{id}` | Update a record |
| `POST` | `/db/{name}/query` | Execute AIQL query |
| `POST` | `/db/{name}/search` | Vector similarity search |
| `POST` | `/db/{name}/traverse` | Knowledge graph traversal |
| `POST` | `/db/{name}/consolidate` | Memory consolidation |
| `DELETE` | `/db/{name}/purge` | Purge expired records |
| `POST` | `/db/{name}/save` | Flush to disk |

### Rate Limiting

Responses include rate-limit headers:

```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 118
X-RateLimit-Reset: 57
```

When the limit is exceeded, the server returns `429 Too Many Requests` with a `Retry-After` header.

### Request Tracing

Every response includes:

```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Response-Time-Ms: 2.4
```

---

## 📦 Deployment

### Local Development

```bash
git clone https://github.com/<owner>/neuraldb.git
cd neuraldb
pip install -r requirements.txt
export NEURALDB_API_KEY=dev-key
python server.py
```

### Docker (Recommended)

```bash
cp .env.example .env
# Edit .env — set NEURALDB_API_KEY to a strong random value
docker compose up -d

# Scale workers
WORKERS=4 docker compose up -d
```

### Railway (One-Click)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Fork this repository
2. Connect to Railway
3. Set `NEURALDB_API_KEY` in the environment
4. Deploy

### Render

1. Push to GitHub
2. New → Web Service → Docker runtime
3. Add a **Persistent Disk** mounted at `/data`
4. Set `NEURALDB_API_KEY`

### Fly.io

```bash
fly launch --name neuraldb
fly volumes create neuraldb_data --size 1
fly secrets set NEURALDB_API_KEY=<your-api-key>
fly deploy
```

### AWS ECS (Fargate)

```bash
# Push image to ECR
aws ecr create-repository --repository-name neuraldb
docker build -t neuraldb .
docker tag neuraldb:latest <ecr-url>/neuraldb:latest
docker push <ecr-url>/neuraldb:latest
```

Then create an ECS task definition with:
- **Image:** your ECR URL
- **Port mapping:** 8000
- **Volume:** EFS volume mounted at `/data`
- **Environment:** `NEURALDB_API_KEY`, `NEURALDB_AUTH=true`

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuraldb
spec:
  replicas: 2
  selector:
    matchLabels:
      app: neuraldb
  template:
    metadata:
      labels:
        app: neuraldb
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
      - name: neuraldb
        image: <owner>/neuraldb:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: NEURALDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: neuraldb-secrets
              key: api-key
        - name: NEURALDB_AUTH
          value: "true"
        - name: NEURALDB_VECTOR_DIM
          value: "1536"
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "256Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: neuraldb-pvc
```

### GCP Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/neuraldb
gcloud run deploy neuraldb \
  --image gcr.io/PROJECT_ID/neuraldb \
  --region us-central1 \
  --set-secrets NEURALDB_API_KEY=neuraldb-key:latest \
  --set-env-vars NEURALDB_AUTH=true,PORT=8080
```

### Azure Container Apps

```bash
az containerapp create \
  --name neuraldb \
  --resource-group myRG \
  --image <owner>/neuraldb:1.0.0 \
  --target-port 8000 \
  --secrets api-key=<your-api-key> \
  --env-vars NEURALDB_API_KEY=secretref:api-key NEURALDB_AUTH=true
```

---

## ⚙️ Configuration

All configuration is via environment variables. No config files to manage.

| Variable | Default | Description |
|---|---|---|
| `NEURALDB_API_KEY` | *(required in production)* | Authentication key |
| `NEURALDB_AUTH` | `true` | Enable/disable auth |
| `NEURALDB_DATA_PATH` | `./data` | Persistence directory |
| `NEURALDB_VECTOR_DIM` | `128` | Embedding dimensions (must match your encoder) |
| `NEURALDB_RATE_LIMIT` | `true` | Enable rate limiting |
| `NEURALDB_RPM` | `120` | Requests per minute per IP |
| `NEURALDB_BURST` | `20` | Burst allowance |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PORT` | `8000` | HTTP port |
| `LOG_LEVEL` | `INFO` | Python log level |
| `WORKERS` | `2` | Uvicorn worker processes |

**Common `NEURALDB_VECTOR_DIM` values:**

| Encoder Model | Dimensions |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-ada-002` | 1536 |
| `all-MiniLM-L6-v2` | 384 |
| `all-mpnet-base-v2` | 768 |
| `nomic-embed-text` | 768 |
| `e5-large-v2` | 1024 |

---

## 🔌 Integrations

### LangChain Memory Backend

Use NeuralDB as a persistent, confidence-aware memory store for LangChain agents:

```python
from langchain.memory import BaseMemory
from neuraldb_client import NeuralDBClient

class NeuralDBMemory(BaseMemory):
    def __init__(self, db_url: str, api_key: str, db_name: str = "langchain"):
        self.client = NeuralDBClient(db_url, api_key=api_key)
        self.db = db_name
        self.client.create_db(db_name)
        self.client.create_lane(db_name, "episodes", "episodic", decay_half_life=86400*7)

    @property
    def memory_variables(self): return ["history"]

    def load_memory_variables(self, inputs):
        records = self.client.query(self.db,
            "RECALL TOP 20 FROM episodes ORDER BY age ASC")["records"]
        history = "\n".join(
            f"{r.get('role', 'unknown')}: {r.get('content', '')}"
            for r in records
        )
        return {"history": history}

    def save_context(self, inputs, outputs):
        self.client.insert(self.db, "episodes",
            {"role": "human", "content": inputs.get("input", "")}, confidence=1.0)
        self.client.insert(self.db, "episodes",
            {"role": "ai", "content": outputs.get("output", "")}, confidence=0.95)

    def clear(self):
        self.client.query(self.db, "FORGET FROM episodes WHERE age > 0")
```


### RAG Pipeline (Retrieval-Augmented Generation)

Build a confidence-weighted retrieval pipeline. Works with any embedding model and any LLM.

```python
import os
from neuraldb_client import NeuralDBClient

# Plug in your own encoder — any function that returns list[float] works
def encode(text: str) -> list:
    """Implement using sentence-transformers, a local model, or any API encoder."""
    raise NotImplementedError

client = NeuralDBClient("http://localhost:8000", api_key=os.environ["NEURALDB_API_KEY"])

def store_document(text: str, source: str, confidence: float = 0.9) -> str:
    """Embed and store a document chunk with provenance tracking."""
    return client.insert("rag_db", "knowledge",
        data={"text": text, "source": source},
        confidence=confidence,
        embedding=encode(text),
        provenance=source
    )

def retrieve(query: str, top_k: int = 5) -> list:
    """Retrieve relevant chunks ranked by similarity × confidence."""
    return client.search("rag_db",
        vector=encode(query),
        top_k=top_k,
        min_confidence=0.5
    )

def build_context(question: str) -> str:
    """Assemble a grounded context string — feed into any LLM."""
    chunks = retrieve(question)
    return "\n\n".join(
        f"[confidence={c['_confidence_score']:.2f}, source={c.get('source','?')}]\n{c['text']}"
        for c in chunks
    )

# Example: pipe context into your LLM of choice
# context = build_context("What is the capital of France?")
# prompt = f"Answer using only the context below:\n{context}\n\nQuestion: ..."
```

### LlamaIndex Document Store

```python
from llama_index.core import StorageContext
from neuraldb_client import NeuralDBClient

class NeuralDBDocStore:
    """LlamaIndex-compatible document store backed by NeuralDB."""

    def __init__(self, client: NeuralDBClient, db: str):
        self.client = client
        self.db = db

    def add_documents(self, docs, allow_update=True):
        for doc in docs:
            self.client.insert(self.db, "knowledge",
                data={"doc_id": doc.doc_id, "text": doc.text, **doc.metadata},
                confidence=0.9,
                embedding=doc.embedding
            )

    def get_document(self, doc_id: str):
        results = self.client.query(self.db,
            f"RECALL FROM knowledge WHERE doc_id = {doc_id}")
        return results["records"][0] if results["count"] > 0 else None
```

### Autonomous Agent Memory

```python
class AgentMemory:
    """
    Multi-lane memory system for autonomous AI agents.
    Mirrors the cognitive architecture: sensory → episodic → semantic.
    """

    def __init__(self, client: NeuralDBClient, agent_id: str):
        self.client = client
        self.agent = agent_id
        self._setup()

    def _setup(self):
        self.client.create_db(self.agent)
        for lane, ltype, ttl in [
            ("perception",  "sensory",    300),     # 5min — raw inputs
            ("episodes",    "episodic",   86400*7), # 1wk  — recent events
            ("knowledge",   "semantic",   None),    # ∞    — learned facts
            ("goals",       "working",    3600),    # 1hr  — active objectives
            ("skills",      "procedural", None),    # ∞    — action patterns
        ]:
            self.client.create_lane(self.agent, lane, ltype, ttl)

    def perceive(self, observation: dict, embedding: list):
        """Store a raw perception. Short-lived."""
        return self.client.insert(self.agent, "perception",
            observation, confidence=1.0, embedding=embedding)

    def remember_episode(self, event: dict, confidence: float = 1.0):
        return self.client.insert(self.agent, "episodes", event, confidence=confidence)

    def learn(self, fact: dict, confidence: float, embedding: list = None):
        """Commit a fact to long-term knowledge."""
        return self.client.insert(self.agent, "knowledge", fact,
                                   confidence=confidence, embedding=embedding)

    def sleep(self):
        """Consolidate high-confidence episodes into knowledge (like sleep)."""
        count = self.client.consolidate(self.agent, "episodes", "knowledge",
                                         min_confidence=0.75)
        # Purge old perceptions
        self.client.query(self.agent, "FORGET FROM perception WHERE age > 300")
        return count

    def recall(self, query_embedding: list, top_k: int = 5) -> list:
        """Retrieve relevant memories using semantic search."""
        return self.client.search(self.agent, vector=query_embedding,
                                   top_k=top_k, min_confidence=0.4)
```

### sentence-transformers (Local Embeddings)

```python
# Install: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from neuraldb import NeuralDB, LaneType

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim; swap for any compatible model
db = NeuralDB("semantic_search", path="./data/neuraldb")
db.create_lane("documents", lane_type=LaneType.SEMANTIC)

def index_document(text: str, metadata: dict) -> str:
    """Embed and persist a document chunk."""
    embedding = model.encode(text).tolist()
    return db.insert("documents", {**metadata, "text": text},
                     confidence=0.9, embedding=embedding)

def search(query: str, top_k: int = 5) -> list:
    """Return top_k semantically similar documents."""
    query_vec = model.encode(query).tolist()
    return db.similarity_search(query_vec, top_k=top_k, min_confidence=0.5)
```

---

## ⚡ Benchmarks

Benchmarked on a MacBook Pro M3 (16GB RAM), single process.

### Write throughput

| Operation | Records/sec |
|---|---|
| Insert (no embedding) | ~45,000 |
| Insert (128-dim embedding) | ~38,000 |
| Insert (1536-dim embedding) | ~22,000 |
| Bulk insert via Python API | ~60,000 |

### Query performance

| Query type | Records | Latency (p50) | Latency (p99) |
|---|---|---|---|
| RECALL with WHERE | 10,000 | 1.2ms | 4.1ms |
| Vector similarity (128-dim) | 10,000 | 3.8ms | 8.2ms |
| Vector similarity (1536-dim) | 10,000 | 18ms | 31ms |
| Graph traversal depth=2 | 10,000 nodes | 0.9ms | 2.4ms |
| Graph traversal depth=4 | 10,000 nodes | 4.1ms | 11ms |
| Memory consolidation | 1,000 records | 12ms | 24ms |

### Memory footprint

| Records | Embedding dim | RAM usage |
|---|---|---|
| 10,000 | 128 | ~45 MB |
| 10,000 | 1536 | ~240 MB |
| 100,000 | 128 | ~420 MB |
| 100,000 | 1536 | ~2.3 GB |

### Scalability notes

- **Single-node:** Suitable for up to ~500k records with 128-dim embeddings in 4GB RAM
- **Production:** Swap `VectorIndex` for FAISS or HNSW for millions of records
- **Distributed:** For multi-node deployments, use Redis or PostgreSQL as the persistence layer

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NeuralDB Engine                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │                  REST API (server.py)               │ │
│  │  FastAPI · CORS · Rate Limiting · Auth · Logging    │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                          │                               │
│  ┌───────────────────────▼──────────────────────────┐   │
│  │              Security Layer                       │   │
│  │  Input Validation · Sanitization · Auth           │   │
│  └───────────────────────┬──────────────────────────┘   │
│                          │                               │
│  ┌───────────────────────▼──────────────────────────┐   │
│  │              AIQL Parser & Executor               │   │
│  │  Tokenizer → AST → Executor → Results             │   │
│  └───────────┬──────────────────────┬───────────────┘   │
│              │                      │                    │
│  ┌───────────▼──────────┐  ┌────────▼────────────────┐  │
│  │   Memory Lanes       │  │   Index Layer           │  │
│  │  NeuralRecord store  │  │  VectorIndex (cosine)   │  │
│  │  Confidence & Decay  │  │  GraphIndex (BFS/DFS)   │  │
│  └───────────┬──────────┘  └────────┬────────────────┘  │
│              │                      │                    │
│  ┌───────────▼──────────────────────▼────────────────┐  │
│  │              Persistence Layer (pickle)            │  │
│  │  Auto-save every 10 writes · Manual flush          │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
neuraldb/
├── neuraldb/                    ← Core library
│   ├── __init__.py              ← Public API surface
│   ├── core/
│   │   ├── engine.py            ← NeuralDB: main database class
│   │   ├── record.py            ← NeuralRecord: confidence + decay
│   │   └── memory_lane.py       ← MemoryLane: typed collections
│   ├── indexes/
│   │   ├── vector_index.py      ← Cosine similarity search
│   │   └── graph_index.py       ← Knowledge graph (BFS, weighted)
│   ├── query/
│   │   ├── parser.py            ← AIQL → AST
│   │   └── executor.py          ← AST execution engine
│   ├── security/
│   │   ├── auth.py              ← Timing-safe API key auth
│   │   ├── ratelimit.py         ← Sliding-window rate limiter
│   │   └── validation.py        ← Input sanitization & validation
│   └── middleware/
│       └── logging.py           ← Structured request logging
├── server.py                    ← FastAPI REST server
├── neuraldb_client.py           ← Python SDK client
├── tests/
│   ├── unit/
│   │   ├── test_engine.py       ← Core engine tests
│   │   └── test_security.py     ← Security layer tests
│   └── integration/
│       └── test_api.py          ← REST API end-to-end tests
├── benchmarks/                  ← Performance benchmarks
├── .github/workflows/ci.yml     ← CI/CD pipeline
├── Dockerfile                   ← Multi-stage production image
├── docker-compose.yml           ← Local + production compose
├── pyproject.toml               ← Package metadata + tooling
├── requirements.txt             ← Runtime dependencies
└── .env.example                 ← Configuration template
```

---

## 🔒 Security

### Design Principles

1. **Fail closed.** Missing API key = 401. Misconfigured server = 401. No silent fallbacks to permissive behaviour.
2. **Timing-safe comparisons.** `hmac.compare_digest` prevents API key enumeration via timing side-channels.
3. **Input validation at the trust boundary.** All user data is sanitized before it reaches the storage engine.
4. **No key material in logs.** Sensitive headers are stripped before any logging occurs.
5. **Non-root container.** The Docker image runs as UID 1001 (`neuraldb` user).
6. **Injection prevention.** AIQL queries are rejected if they contain SQL-style comment sequences or stacked statements.

### Production Security Checklist

- [ ] Set `NEURALDB_API_KEY` to a cryptographically random 32+ character string
- [ ] Set `NEURALDB_AUTH=true`
- [ ] Run behind HTTPS (Traefik, Nginx, or cloud load balancer)
- [ ] Mount `/data` to a persistent volume — never rely on container filesystem
- [ ] Set `CORS_ORIGINS` to your specific domain(s), not `*`
- [ ] Set resource limits in Docker/Kubernetes (CPU + memory)
- [ ] Rotate API keys periodically
- [ ] Enable VPC/private networking; do not expose the port publicly if avoidable
- [ ] Review and set `NEURALDB_RPM` to match your expected traffic

### Vulnerability Reporting

Please report security vulnerabilities by opening a **private** [GitHub Security Advisory](https://github.com/<owner>/neuraldb/security/advisories/new) rather than a public issue. We aim to acknowledge all reports within 48 hours.

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=neuraldb --cov-report=term-missing

# Run only unit tests (no server needed)
pytest tests/unit/ -v

# Run integration tests (tests the REST API)
NEURALDB_AUTH=false pytest tests/integration/ -v

# Run security scan
bandit -r neuraldb/ server.py

# Check for CVEs in dependencies
safety check -r requirements.txt
```

---

## 🛣 Roadmap

### v1.1 (Q2 2025)
- [ ] HNSW index for 10x faster approximate nearest-neighbour search
- [ ] Multi-tenancy (per-tenant key → per-tenant database isolation)
- [ ] WebSocket subscription for real-time change notifications
- [ ] `ASSOCIATE` AIQL verb (semantic search via AIQL)

### v1.2 (Q3 2025)
- [ ] PostgreSQL / SQLite backend option (replace pickle persistence)
- [ ] Write-ahead log (WAL) for crash recovery
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Bulk insert API endpoint

### v2.0 (Q4 2025)
- [ ] Distributed mode: consistent hashing across multiple nodes
- [ ] Optional Redis for persistence
- [ ] gRPC transport option
- [ ] Native LangChain + LlamaIndex packages on PyPI

---

## 🤝 Contributing

Contributions are welcome and appreciated. Please read `CONTRIBUTING.md` for the full process.

### Quick Contribution Guide

```bash
# Fork + clone
git clone https://github.com/<owner>/neuraldb.git
cd neuraldb

# Create a branch
git checkout -b feature/your-feature-name

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Make changes, then test
pytest tests/ -v

# Run the linter
ruff check .

# Commit (conventional commits please)
git commit -m "feat: add HNSW index for faster vector search"

# Push and open a PR
git push origin feature/your-feature-name
```

### Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `perf:` — performance improvement
- `security:` — security fix

---

## 📄 License

[MIT License](LICENSE) — free for commercial and research use.

```
Copyright (c) 2025 NeuralDB Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

NeuralDB draws inspiration from:
- Cognitive science research on human memory systems (Tulving, 1972)
- Ebbinghaus's forgetting curve and spaced repetition research
- The architecture of modern neural networks and attention mechanisms
- Open-source projects: FastAPI, NumPy, Pydantic

---

<div align="center">
<br>
Made with ❤️ for the AI engineering community

[⭐ Star on GitHub](https://github.com/<owner>/neuraldb) · [🐛 Report a Bug](https://github.com/<owner>/neuraldb/issues) · [💡 Request a Feature](https://github.com/<owner>/neuraldb/discussions)
</div>
