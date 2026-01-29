# chatbot.py
"""
TechBot Pro - Advanced NLP Tech Assistant
Fixed version with improved:
- Intent detection (catches "explain X", "what is X", etc.)
- Better keyword expansion and entity grounding
- Improved prompts for flan-t5 model
- Better fallback responses with built-in knowledge base
"""

import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import requests
import feedparser

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ChatbotBase import ChatbotBase

# Use flan-t5-large for better quality (flan-t5-small gives fragmented answers)
DEFAULT_BASE_MODEL = "google/flan-t5-large"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(s: str) -> str:
    s = re.sub(r"<[^<]+?>", "", s or "")
    s = re.sub(r"&nbsp;+", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate(s: Optional[str], max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _strip_links_and_bullets(text: str) -> str:
    """Keep context clean so the model doesn't start speaking in headlines."""
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("-", "•", "*")):
            continue
        if "http://" in s or "https://" in s:
            continue
        if s.lower().startswith(("rules:", "user level:", "sources:", "why it matters:")):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


class TechBot(ChatbotBase):
    """
    TechBot Pro - Fixed Version
    
    Key improvements:
    1. Better intent detection for questions like "explain X", "what is X"
    2. Expanded knowledge base for common tech topics
    3. Better RSS filtering for relevant news
    4. Improved prompts that work better with flan-t5
    5. Rich fallback answers when LLM fails
    """

    def __init__(
        self,
        name: str = "TechBot Pro",
        db_path: str = "techbot.db",
        base_model: str = DEFAULT_BASE_MODEL,
        hf_cache_dir: Optional[str] = None,
        max_history_turns: int = 6,
    ):
        super().__init__(name)
        self.db_path = db_path
        self.max_history_turns = max_history_turns

        # Load knowledge bases
        self.keyword_expansion = self._load_keyword_expansion()
        self.entity_grounding = self._load_entity_grounding()
        self.learning_resources = self._load_learning_resources()
        self.tech_knowledge = self._load_tech_knowledge()  # NEW: Built-in knowledge

        self._init_database()
        self._init_llm(base_model=base_model, hf_cache_dir=hf_cache_dir)

        self._news_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

        print(f"✅ {self.name} initialized successfully!")
        print("   - General tech Q&A (default)")
        print("   - UPDATE MODE for news/updates queries")
        print("   - Profile + progress tracking enabled")
        print(f"   - Model: {self.llm_name if getattr(self,'llm_ready', False) else 'NOT LOADED'}\n")

    # =========================
    # BUILT-IN TECH KNOWLEDGE (NEW)
    # =========================
    def _load_tech_knowledge(self) -> Dict[str, str]:
        """
        Comprehensive built-in knowledge base for common tech topics.
        This ensures the bot can answer even when LLM produces poor output.
        """
        return {
            # OpenAI / ChatGPT
            "chatgpt": """**ChatGPT** is an AI chatbot developed by OpenAI, launched in November 2022.

**Key Features:**
- Built on GPT (Generative Pre-trained Transformer) architecture
- Can understand and generate human-like text
- Supports conversations, coding help, writing, analysis, and more
- Available in free (GPT-3.5) and paid (GPT-4, GPT-4o) versions

**Capabilities:**
- Answer questions and explain concepts
- Write and debug code in multiple languages
- Create content (essays, emails, stories)
- Analyze data and documents
- Generate images (with DALL-E integration)

**Recent Updates (2024-2025):**
- GPT-4o: Faster, multimodal (text, vision, audio)
- Memory feature: Remembers past conversations
- Custom GPTs: Users can create specialized assistants
- Voice mode: Real-time voice conversations""",

            "openai": """**OpenAI** is an AI research company founded in 2015.

**Key Products:**
- **ChatGPT**: Conversational AI assistant (100M+ users)
- **GPT-4/GPT-4o**: Advanced language models
- **DALL-E**: AI image generation
- **Whisper**: Speech-to-text model
- **Sora**: AI video generation (2024)
- **API Platform**: For developers to build AI apps

**Leadership:** Sam Altman (CEO), with major investment from Microsoft (~$13B)

**Recent Developments:**
- Launched GPT-4o (omni) with multimodal capabilities
- Released GPT-4 Turbo with 128K context window
- Introduced Custom GPTs and GPT Store
- Working on AGI (Artificial General Intelligence) research""",

            "gpt": """**GPT (Generative Pre-trained Transformer)** is OpenAI's family of large language models.

**Evolution:**
- **GPT-1 (2018)**: 117M parameters, proved transformer pre-training works
- **GPT-2 (2019)**: 1.5B parameters, impressive text generation
- **GPT-3 (2020)**: 175B parameters, few-shot learning breakthrough
- **GPT-3.5 (2022)**: Powers free ChatGPT
- **GPT-4 (2023)**: Multimodal, much more capable reasoning
- **GPT-4o (2024)**: Faster, native multimodal (text/audio/vision)

**How It Works:**
1. Pre-trained on massive text data from the internet
2. Fine-tuned with human feedback (RLHF)
3. Uses transformer architecture with attention mechanism
4. Generates text by predicting next tokens""",

            # AI General
            "ai": """**Artificial Intelligence (AI)** is technology that enables machines to perform tasks requiring human-like intelligence.

**Types of AI:**
- **Narrow AI**: Specialized for specific tasks (current AI)
- **General AI (AGI)**: Human-level intelligence across all domains (future goal)
- **Superintelligence**: Beyond human intelligence (theoretical)

**Key AI Technologies:**
- **Machine Learning**: Systems that learn from data
- **Deep Learning**: Neural networks with many layers
- **NLP**: Understanding and generating human language
- **Computer Vision**: Understanding images and video
- **Reinforcement Learning**: Learning through trial and error

**Current AI Applications:**
- Chatbots (ChatGPT, Claude, Gemini)
- Image generation (DALL-E, Midjourney, Stable Diffusion)
- Code assistants (GitHub Copilot, Cursor)
- Autonomous vehicles
- Medical diagnosis
- Recommendation systems""",

            "llm": """**Large Language Models (LLMs)** are AI models trained on massive text data to understand and generate language.

**How They Work:**
1. Trained on billions of text tokens (web pages, books, code)
2. Learn patterns, grammar, facts, and reasoning
3. Generate text by predicting the next word/token
4. Fine-tuned for specific tasks (chat, coding, etc.)

**Popular LLMs:**
- **GPT-4/GPT-4o** (OpenAI) - Powers ChatGPT
- **Claude 3** (Anthropic) - Known for safety and long context
- **Gemini** (Google) - Multimodal capabilities
- **Llama 3** (Meta) - Open-source
- **Mistral** (Mistral AI) - Efficient open-source models

**Key Concepts:**
- **Parameters**: Model size (7B, 70B, 175B, etc.)
- **Context Window**: How much text it can process at once
- **Tokens**: Pieces of text the model works with
- **Fine-tuning**: Adapting model for specific tasks
- **RLHF**: Training with human feedback""",

            "rag": """**RAG (Retrieval-Augmented Generation)** combines search with AI generation for accurate, sourced answers.

**How RAG Works:**
1. **Query**: User asks a question
2. **Retrieve**: Search a knowledge base for relevant documents
3. **Augment**: Add retrieved context to the prompt
4. **Generate**: LLM generates answer using the context

**Benefits:**
- ✅ Reduces hallucinations (grounded in real documents)
- ✅ Provides citations and sources
- ✅ Works with private/updated data
- ✅ No need to retrain the model

**Common RAG Stack:**
- **Embeddings**: OpenAI, Cohere, or open-source models
- **Vector DB**: Pinecone, Weaviate, Chroma, Qdrant
- **LLM**: GPT-4, Claude, Llama, etc.
- **Framework**: LangChain, LlamaIndex

**Use Cases:**
- Company knowledge bases
- Customer support bots
- Research assistants
- Document Q&A systems""",

            # Hardware
            "gpu": """**GPU (Graphics Processing Unit)** is a processor designed for parallel computing, essential for AI and gaming.

**Why GPUs Matter for AI:**
- Can perform thousands of calculations simultaneously
- Neural network training requires massive parallel math
- Modern LLMs need powerful GPUs (or clusters of them)

**Key GPU Companies:**
- **NVIDIA**: Dominates AI GPUs (H100, A100, RTX series)
- **AMD**: Growing AI presence (MI300X)
- **Intel**: Arc GPUs, Gaudi AI accelerators

**NVIDIA AI GPUs:**
- **H100**: Current top AI training GPU (~$30,000)
- **H200**: Next-gen with more memory
- **A100**: Previous generation, still widely used
- **RTX 4090**: Consumer GPU, good for local AI

**GPU Memory (VRAM) Matters:**
- 7B parameter model needs ~14GB VRAM
- 70B parameter model needs ~140GB VRAM
- Training requires even more memory""",

            "nvidia": """**NVIDIA** is the leading GPU and AI chip company.

**Key Products:**
- **Data Center GPUs**: H100, H200, A100 (for AI training)
- **Consumer GPUs**: RTX 4090, 4080, 4070 series
- **AI Platforms**: DGX systems, CUDA software
- **Automotive**: DRIVE platform for self-driving cars

**Why NVIDIA Dominates AI:**
- **CUDA**: Software ecosystem for GPU computing
- **10+ years** head start in AI/ML optimization
- **Best performance** for training large models
- **Strong ecosystem**: Libraries, tools, developer support

**Recent News (2024-2025):**
- Blackwell architecture announced (next-gen)
- Stock surged due to AI demand
- Expanding into robotics and autonomous systems
- Partnership with major cloud providers
- Revenue tripled due to AI boom""",

            # Cloud & Infrastructure
            "cloud": """**Cloud Computing** provides on-demand computing resources over the internet.

**Major Cloud Providers:**
- **AWS (Amazon)**: Largest market share (~32%)
- **Azure (Microsoft)**: Strong enterprise presence (~23%)
- **GCP (Google)**: Leading in AI/ML services (~10%)

**Cloud Service Types:**
- **IaaS**: Virtual machines, storage (EC2, S3)
- **PaaS**: Managed platforms (Heroku, Cloud Run)
- **SaaS**: Complete applications (Salesforce, Office 365)

**Key Cloud Services:**
- Compute (VMs, containers, serverless)
- Storage (object, block, file)
- Databases (SQL, NoSQL, data warehouses)
- AI/ML (training, inference, APIs)
- Networking (VPCs, load balancers, CDNs)

**Cloud for AI:**
- GPU instances for training
- Managed ML platforms (SageMaker, Vertex AI)
- Pre-built AI APIs (Vision, Language, Speech)""",

            "kubernetes": """**Kubernetes (K8s)** is an open-source container orchestration platform.

**What It Does:**
- Automates deployment of containerized applications
- Handles scaling, load balancing, self-healing
- Manages container networking and storage

**Key Concepts:**
- **Pod**: Smallest deployable unit (one or more containers)
- **Deployment**: Manages replicas of pods
- **Service**: Exposes pods to network traffic
- **Ingress**: Routes external traffic to services
- **ConfigMap/Secret**: Store configuration and sensitive data

**Why Use Kubernetes:**
- ✅ Automatic scaling based on load
- ✅ Self-healing (restarts failed containers)
- ✅ Rolling updates with zero downtime
- ✅ Works across cloud providers

**Managed Kubernetes:**
- EKS (AWS), GKE (Google), AKS (Azure)""",

            "docker": """**Docker** is a platform for building and running containerized applications.

**What Are Containers:**
- Lightweight, isolated environments for apps
- Include code, runtime, libraries, and dependencies
- Run consistently across different machines

**Key Docker Concepts:**
- **Image**: Blueprint for containers (built from Dockerfile)
- **Container**: Running instance of an image
- **Dockerfile**: Instructions to build an image
- **Docker Hub**: Registry for sharing images
- **Docker Compose**: Define multi-container apps

**Basic Commands:**
```bash
docker build -t myapp .     # Build image
docker run -p 3000:3000 myapp  # Run container
docker ps                   # List running containers
docker-compose up           # Start multi-container app
```

**Benefits:**
- ✅ "Works on my machine" solved
- ✅ Fast startup (seconds vs minutes for VMs)
- ✅ Consistent dev/prod environments
- ✅ Easy scaling and deployment""",

            # Programming
            "python": """**Python** is a popular, beginner-friendly programming language.

**Why Python is Popular:**
- Easy to read and write syntax
- Huge ecosystem of libraries
- Dominant in AI/ML, data science, automation
- Great for beginners and experts alike

**Key Python Libraries:**
- **AI/ML**: TensorFlow, PyTorch, scikit-learn, transformers
- **Data**: pandas, numpy, matplotlib, seaborn
- **Web**: Django, Flask, FastAPI
- **Automation**: requests, selenium, beautifulsoup

**Python for AI:**
```python
# Example: Using transformers library
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```

**Learning Resources:**
- Python.org official tutorial
- Automate the Boring Stuff (free book)
- freeCodeCamp Python course
- Real Python tutorials""",

            "javascript": """**JavaScript** is the language of the web, running in browsers and servers.

**Where JavaScript Runs:**
- **Browser**: Dynamic web pages, React/Vue/Angular apps
- **Server**: Node.js, Deno, Bun
- **Mobile**: React Native, Ionic
- **Desktop**: Electron (VS Code, Discord)

**Key Frameworks:**
- **React**: Most popular UI library (Meta)
- **Vue**: Progressive framework, easy to learn
- **Angular**: Full framework by Google
- **Next.js**: React framework with SSR
- **Node.js**: JavaScript runtime for servers

**Modern JavaScript (ES6+):**
```javascript
// Arrow functions
const add = (a, b) => a + b;

// Async/await
const data = await fetch('/api/data');

// Destructuring
const { name, age } = user;
```

**JavaScript for AI:**
- TensorFlow.js (ML in browser)
- Langchain.js (LLM applications)
- AI SDK by Vercel (React AI apps)""",

            # Companies
            "microsoft": """**Microsoft** is a technology giant known for Windows, Office, Azure, and AI.

**Key Products:**
- **Windows**: Dominant desktop OS
- **Microsoft 365**: Office, Teams, Outlook
- **Azure**: #2 cloud platform
- **GitHub**: Largest code hosting platform
- **LinkedIn**: Professional networking
- **Xbox**: Gaming division

**Microsoft & AI:**
- $13B+ invested in OpenAI
- **Copilot**: AI assistant across products
- **GitHub Copilot**: AI coding assistant
- **Azure OpenAI**: GPT-4 API access
- **Bing Chat**: ChatGPT-powered search

**Recent Developments:**
- Copilot integrated into Windows 11
- AI features in Office apps
- Major cloud AI infrastructure investments""",

            "google": """**Google** is a technology leader in search, cloud, and AI.

**Key Products:**
- **Search**: 90%+ market share
- **Android**: Most used mobile OS
- **Chrome**: Most popular browser
- **YouTube**: Largest video platform
- **Google Cloud (GCP)**: #3 cloud provider
- **Workspace**: Gmail, Docs, Drive, Meet

**Google AI:**
- **Gemini**: Latest multimodal AI model
- **Bard**: AI chatbot (now Gemini)
- **DeepMind**: AI research lab (AlphaGo, AlphaFold)
- **TensorFlow**: Popular ML framework
- **TPUs**: Custom AI chips

**Recent AI Developments:**
- Gemini Ultra competes with GPT-4
- AI integrated into Search (SGE)
- Workspace AI features (Duet AI)
- NotebookLM for document analysis""",

            "apple": """**Apple** is known for premium hardware and integrated ecosystems.

**Key Products:**
- **iPhone**: Premium smartphone
- **Mac**: Computers with Apple Silicon
- **iPad**: Tablets
- **Apple Watch**: Smartwatch leader
- **AirPods**: Wireless earbuds
- **Vision Pro**: Mixed reality headset

**Apple Silicon:**
- M1, M2, M3, M4 chips
- ARM-based, very power efficient
- Great for local AI/ML tasks
- Unified memory architecture

**Apple & AI:**
- On-device ML (Core ML)
- Siri improvements
- Apple Intelligence (iOS 18+)
- Privacy-focused AI approach
- Partnership with OpenAI announced""",

            "meta": """**Meta** (formerly Facebook) focuses on social media and the metaverse.

**Key Products:**
- **Facebook**: Largest social network
- **Instagram**: Photo/video sharing
- **WhatsApp**: Messaging (2B+ users)
- **Messenger**: Facebook messaging
- **Quest**: VR headsets
- **Threads**: Twitter/X competitor

**Meta AI:**
- **Llama**: Open-source LLMs (Llama 2, Llama 3)
- **FAIR**: AI research lab
- **PyTorch**: Popular ML framework (originally Meta)
- **Meta AI**: Assistant across apps

**Why Llama Matters:**
- Free and open-source
- Competitive with GPT-3.5
- Can run locally
- Used by many companies and researchers""",

            # Cybersecurity
            "cybersecurity": """**Cybersecurity** protects systems, networks, and data from digital attacks.

**Common Threats:**
- **Phishing**: Fake emails/sites to steal credentials
- **Ransomware**: Encrypts data, demands payment
- **Malware**: Viruses, trojans, spyware
- **DDoS**: Overwhelms systems with traffic
- **Zero-day**: Exploits unknown vulnerabilities

**Security Best Practices:**
- ✅ Use strong, unique passwords
- ✅ Enable two-factor authentication (2FA)
- ✅ Keep software updated
- ✅ Be cautious with links and attachments
- ✅ Regular backups

**Security Careers:**
- Security Analyst
- Penetration Tester
- Security Engineer
- CISO (Chief Information Security Officer)

**Certifications:**
- CompTIA Security+
- CEH (Certified Ethical Hacker)
- CISSP (advanced)""",

            # Web Development
            "react": """**React** is a JavaScript library for building user interfaces.

**Key Concepts:**
- **Components**: Reusable UI pieces
- **JSX**: HTML-like syntax in JavaScript
- **Props**: Pass data to components
- **State**: Component's internal data
- **Hooks**: useState, useEffect, etc.

**Example Component:**
```jsx
function Counter() {
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount(count + 1)}>
      Clicked {count} times
    </button>
  );
}
```

**React Ecosystem:**
- **Next.js**: Full-stack React framework
- **React Router**: Client-side routing
- **Redux/Zustand**: State management
- **React Query**: Server state management

**Why React:**
- Most popular UI library
- Large ecosystem and community
- Used by Meta, Netflix, Airbnb
- React Native for mobile apps""",

            "nextjs": """**Next.js** is a React framework for production web applications.

**Key Features:**
- **Server-Side Rendering (SSR)**: Fast initial loads
- **Static Site Generation (SSG)**: Pre-built pages
- **App Router**: File-based routing
- **API Routes**: Backend endpoints
- **Server Components**: Render on server

**Project Structure:**
```
app/
  page.tsx        # Home page
  about/page.tsx  # /about route
  api/route.ts    # API endpoint
components/
  Header.tsx
```

**Why Next.js:**
- Best React production framework
- Great developer experience
- Optimized performance
- Easy deployment (especially on Vercel)
- Full-stack capabilities""",

            # Databases
            "sql": """**SQL (Structured Query Language)** is used to manage relational databases.

**Basic SQL Commands:**
```sql
-- Create table
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100)
);

-- Insert data
INSERT INTO users (name, email) 
VALUES ('John', 'john@example.com');

-- Query data
SELECT * FROM users WHERE name = 'John';

-- Update data
UPDATE users SET name = 'Jane' WHERE id = 1;

-- Delete data
DELETE FROM users WHERE id = 1;
```

**Popular SQL Databases:**
- PostgreSQL (powerful, open-source)
- MySQL (popular, open-source)
- SQLite (lightweight, embedded)
- SQL Server (Microsoft)

**SQL vs NoSQL:**
- SQL: Structured data, relationships, ACID
- NoSQL: Flexible schema, horizontal scaling""",

            "api": """**API (Application Programming Interface)** allows different software to communicate.

**Types of APIs:**
- **REST**: Most common, uses HTTP methods
- **GraphQL**: Query exactly what you need
- **WebSocket**: Real-time bidirectional
- **gRPC**: High-performance, binary

**REST API Example:**
```
GET    /api/users      # List users
GET    /api/users/1    # Get user 1
POST   /api/users      # Create user
PUT    /api/users/1    # Update user 1
DELETE /api/users/1    # Delete user 1
```

**API Authentication:**
- API Keys: Simple, include in header
- OAuth: User authorization
- JWT: Token-based authentication

**Popular API Tools:**
- Postman: API testing
- Swagger/OpenAPI: API documentation
- curl: Command-line requests""",

            # Trending Topics
            "agents": """**AI Agents** are autonomous AI systems that can perform tasks by using tools and making decisions.

**How Agents Work:**
1. Receive a goal/task from user
2. Plan steps to achieve the goal
3. Use tools (search, code, APIs) to execute
4. Observe results and adjust
5. Continue until goal is achieved

**Agent Capabilities:**
- Browse the web
- Write and execute code
- Query databases
- Send emails
- Create documents
- Book appointments

**Agent Frameworks:**
- LangChain Agents
- AutoGPT
- CrewAI
- OpenAI Assistants API

**Examples:**
- Research assistant that finds and summarizes info
- Coding agent that writes and tests code
- Customer service agent that handles tickets""",

            "fine-tuning": """**Fine-tuning** adapts a pre-trained AI model to specific tasks or domains.

**Why Fine-tune:**
- Better performance on your specific use case
- Teach the model your company's style/knowledge
- Reduce costs (smaller model can match larger)
- Improve accuracy for domain-specific tasks

**Fine-tuning Methods:**
- **Full Fine-tuning**: Update all model parameters
- **LoRA**: Low-rank adaptation (efficient)
- **QLoRA**: Quantized LoRA (even more efficient)
- **RLHF**: Reinforcement Learning from Human Feedback

**Steps to Fine-tune:**
1. Prepare training data (instruction/response pairs)
2. Choose base model (Llama, Mistral, etc.)
3. Configure training (learning rate, epochs)
4. Train on your data
5. Evaluate and iterate

**Platforms for Fine-tuning:**
- OpenAI Fine-tuning API
- Hugging Face + Transformers
- AWS SageMaker
- Google Vertex AI""",

            "vector-database": """**Vector Databases** store and search embeddings for AI applications.

**What Are Embeddings:**
- Numerical representations of text, images, etc.
- Similar items have similar vectors
- Enable semantic search (meaning, not keywords)

**How Vector Search Works:**
1. Convert query to embedding
2. Find nearest neighbors in vector space
3. Return most similar items

**Popular Vector Databases:**
- **Pinecone**: Managed, easy to use
- **Weaviate**: Open-source, GraphQL API
- **Chroma**: Simple, Python-native
- **Qdrant**: Fast, open-source
- **Milvus**: Scalable, open-source
- **pgvector**: PostgreSQL extension

**Use Cases:**
- RAG (Retrieval-Augmented Generation)
- Semantic search
- Recommendation systems
- Image similarity search
- Duplicate detection""",
        }

    # =========================
    # DB
    # =========================
    def _init_database(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                knowledge_level TEXT DEFAULT 'beginner',
                interests TEXT DEFAULT '[]',
                goals TEXT DEFAULT '[]',
                created_at TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_text TEXT,
                bot_text TEXT,
                ts TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT '',
                last_updated TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                resource TEXT,
                status TEXT DEFAULT 'suggested',
                ts TEXT
            )
            """
        )
        self.conn.commit()

    # =========================
    # PROFILE (run.py depends on these)
    # =========================
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        self.cursor.execute(
            "SELECT user_id, knowledge_level, interests, goals, created_at FROM users WHERE user_id=?",
            (user_id,),
        )
        row = self.cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "knowledge_level": row[1],
                "interests": json.loads(row[2] or "[]"),
                "goals": json.loads(row[3] or "[]"),
                "created_at": row[4],
            }
        return {
            "user_id": user_id,
            "knowledge_level": "beginner",
            "interests": [],
            "goals": [],
            "created_at": _now_iso(),
        }

    def set_user_profile(
        self,
        user_id: str,
        level: Optional[str] = None,
        interests: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
    ) -> str:
        current = self.get_user_profile(user_id)
        if level:
            current["knowledge_level"] = level
        if interests is not None:
            current["interests"] = interests
        if goals is not None:
            current["goals"] = goals

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO users (user_id, knowledge_level, interests, goals, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                current["knowledge_level"],
                json.dumps(current["interests"]),
                json.dumps(current["goals"]),
                current["created_at"],
            ),
        )
        self.conn.commit()
        self.context["current_user"] = user_id
        return f"✅ Profile updated for {user_id}"

    # =========================
    # PROGRESS (run.py depends on these)
    # =========================
    def track_resource(self, user_id: str, topic: str, resource: str, status: str = "suggested"):
        self.cursor.execute(
            "INSERT INTO progress (user_id, topic, resource, status, ts) VALUES (?, ?, ?, ?, ?)",
            (user_id, topic, resource, status, _now_iso()),
        )
        self.conn.commit()

    def mark_resource_done(self, user_id: str, resource_contains: str) -> str:
        k = (resource_contains or "").strip().lower()
        if not k:
            return "❌ Usage: done <part of resource name>"
        self.cursor.execute(
            "UPDATE progress SET status='done' WHERE user_id=? AND LOWER(resource) LIKE ?",
            (user_id, f"%{k}%"),
        )
        self.conn.commit()
        if self.cursor.rowcount == 0:
            return "⚠️ I couldn't find a suggested resource matching that."
        return f"✅ Marked {self.cursor.rowcount} resource(s) as done."

    def get_progress_report(self, user_id: str) -> str:
        self.cursor.execute(
            """
            SELECT topic, resource, status, ts
            FROM progress
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user_id,),
        )
        rows = self.cursor.fetchall()
        if not rows:
            return "No progress tracked yet. Ask for resources and then mark them with: done <keyword>"

        lines = ["📈 Progress (latest 20):"]
        for topic, resource, status, ts in rows:
            lines.append(f"- [{status}] ({topic}) {resource} — {ts}")
        return "\n".join(lines)

    # =========================
    # MEMORY
    # =========================
    def _get_memory_summary(self, user_id: str) -> str:
        self.cursor.execute("SELECT summary FROM memories WHERE user_id=?", (user_id,))
        row = self.cursor.fetchone()
        return _truncate((row[0] if row and row[0] else ""), 260)

    def _set_memory_summary(self, user_id: str, summary: str):
        summary = _truncate(summary, 260)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO memories (user_id, summary, last_updated)
            VALUES (?, ?, ?)
            """,
            (user_id, summary.strip(), _now_iso()),
        )
        self.conn.commit()

    def _get_recent_dialogue(self, user_id: str, limit: int = 5) -> List[Dict[str, str]]:
        self.cursor.execute(
            "SELECT user_text, bot_text FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = list(reversed(self.cursor.fetchall()))
        out = [{"user": r[0], "bot": r[1]} for r in rows]
        for t in out:
            t["user"] = _truncate(t["user"], 140)
            t["bot"] = _truncate(_strip_links_and_bullets(t["bot"]), 160)
        return out

    def _update_memory_summary(self, user_id: str):
        if not getattr(self, "llm_ready", False):
            return
        turns = self._get_recent_dialogue(user_id, limit=self.max_history_turns)
        if not turns:
            return

        transcript = "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in turns])
        transcript = _truncate(transcript, 700)

        prompt = f"""Summarize the user's tech interests in 3 short bullets.

Conversation:
{transcript}

Bullets:""".strip()

        mem = self._llm_generate(prompt, max_new_tokens=90, mode="safe")
        mem = self._clean_llm_output(mem)
        if mem and len(mem) > 20:
            self._set_memory_summary(user_id, mem)

    def _save_turn(self, user_id: str, user_text: str, bot_text: str):
        self.cursor.execute(
            "INSERT INTO conversations (user_id, user_text, bot_text, ts) VALUES (?, ?, ?, ?)",
            (user_id, user_text, bot_text, _now_iso()),
        )
        self.conn.commit()
        self._update_memory_summary(user_id)

    # =========================
    # GROUNDING / KEYWORDS / RESOURCES
    # =========================
    def _load_keyword_expansion(self) -> Dict[str, List[str]]:
        return {
            "openai": ["chatgpt", "gpt", "open ai", "oai", "gpt-4", "gpt-5", "dall-e", "sora"],
            "chatgpt": ["openai", "gpt", "chat gpt", "gpt-4", "gpt-4o"],
            "gpt": ["openai", "chatgpt", "gpt-4", "gpt-5", "generative pre-trained"],
            "nvidia": ["gpu", "cuda", "rtx", "h100", "h200", "blackwell", "geforce"],
            "ai": ["artificial intelligence", "machine learning", "llm", "genai", "ml", "deep learning"],
            "llm": ["large language model", "gpt", "claude", "llama", "gemini", "language model"],
            "cloud": ["aws", "azure", "gcp", "kubernetes", "docker", "cloud computing"],
            "cybersecurity": ["infosec", "malware", "ransomware", "zero-day", "security", "hacking"],
            "rag": ["retrieval augmented generation", "retrieval-augmented", "vector database"],
            "microsoft": ["windows", "azure", "copilot", "office", "github"],
            "google": ["alphabet", "android", "chrome", "gemini", "bard", "deepmind"],
            "apple": ["iphone", "mac", "ios", "macos", "vision pro", "apple silicon"],
            "meta": ["facebook", "instagram", "whatsapp", "llama", "metaverse"],
            "agents": ["ai agents", "autonomous agents", "agentic", "autogpt"],
        }

    def _expand_keywords(self, text: str) -> List[str]:
        t = (text or "").lower()
        keys = set(re.findall(r"[a-z0-9\-\+\.]+", t))
        for phrase, ex in self.keyword_expansion.items():
            if phrase in t:
                keys.add(phrase)
                keys.update(ex)
            # Also check if any expansion term is in the text
            for exp_term in ex:
                if exp_term in t:
                    keys.add(phrase)
                    keys.update(ex)
                    break
        return sorted(set([k for k in keys if len(k) >= 2]))

    def _load_entity_grounding(self) -> Dict[str, str]:
        return {
            "openai": "OpenAI is an AI research and product company known for ChatGPT and GPT models.",
            "chatgpt": "ChatGPT is a conversational AI assistant product from OpenAI, launched in November 2022.",
            "gpt": "GPT (Generative Pre-trained Transformer) is a family of large language models from OpenAI.",
            "nvidia": "NVIDIA is a semiconductor company best known for GPUs used in gaming and AI.",
            "apple": "Apple builds iPhone, iOS, Mac, and macOS.",
            "microsoft": "Microsoft builds Windows, Azure cloud, and Copilot products.",
            "google": "Google builds Search, Android, Google Cloud, and Gemini AI models.",
            "meta": "Meta (formerly Facebook) builds social platforms and open-source AI like Llama.",
            "rag": "RAG (Retrieval-Augmented Generation) combines retrieval with LLMs for accurate answers.",
            "llm": "LLM (Large Language Model) is an AI model trained on text to generate human-like responses.",
            "ai": "AI (Artificial Intelligence) enables machines to perform tasks requiring human-like intelligence.",
            "agents": "AI Agents are autonomous systems that can use tools and make decisions to achieve goals.",
        }

    def _grounding_lines(self, question: str) -> str:
        q = (question or "").lower()
        hits = []
        for k, v in self.entity_grounding.items():
            if re.search(rf"\b{re.escape(k)}\b", q):
                hits.append(f"- {v}")
        return "\n".join(hits)

    def _load_learning_resources(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "beginner": {
                "ai": ["Coursera: AI For Everyone (Andrew Ng)", "YouTube: 3Blue1Brown – Neural Networks", "fast.ai – Practical Deep Learning for Coders"],
                "cloud": ["AWS Cloud Practitioner free courses", "Azure Fundamentals (AZ-900) roadmap", "Google Cloud Skills Boost"],
                "programming": ["freeCodeCamp – Python track", "Python Crash Course (Eric Matthes)", "Codecademy Python course"],
                "web": ["freeCodeCamp – Web Development", "The Odin Project", "MDN Web Docs"],
            },
            "intermediate": {
                "ai": ["fast.ai – Practical Deep Learning", "Build: RAG chatbot with embeddings + vector DB", "Hugging Face NLP course"],
                "cloud": ["Build: Docker + Kubernetes mini project", "Learn: Terraform basics + deploy app", "AWS Solutions Architect prep"],
                "programming": ["Clean Code by Robert Martin", "Design Patterns (Gang of Four)", "LeetCode practice"],
                "web": ["Next.js documentation", "Full-stack project with database", "React Query / TanStack"],
            },
            "advanced": {
                "ai": ["Fine-tune with LoRA + evaluation harness", "Build: agent + tools + monitoring", "Read: Attention Is All You Need paper"],
                "cloud": ["Kubernetes operators + autoscaling deep dive", "Observability: OpenTelemetry + tracing", "Multi-cloud architecture"],
                "programming": ["System design interviews", "Distributed systems concepts", "Contributing to open source"],
                "web": ["Performance optimization", "Micro-frontends architecture", "Edge computing patterns"],
            },
        }

    def _get_personalized_resources(self, profile: Dict[str, Any], keywords: List[str], topic_hint: str) -> List[str]:
        level = profile.get("knowledge_level", "beginner")
        bucket = self.learning_resources.get(level, self.learning_resources["beginner"])

        wants = set([k.lower() for k in keywords])
        wants.update([topic_hint.lower()])

        interests = [x.lower() for x in (profile.get("interests") or [])]
        wants.update(interests)

        candidates: List[str] = []
        for topic, arr in bucket.items():
            if topic in wants or any(w in topic for w in wants):
                candidates.extend(arr)

        if not candidates:
            candidates = sum(bucket.values(), [])

        out, seen = [], set()
        for r in candidates:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # =========================
    # LLM
    # =========================
    def _init_llm(self, base_model: str, hf_cache_dir: Optional[str]):
        self.llm_name = base_model
        try:
            print("🔄 Loading model:", base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model, cache_dir=hf_cache_dir)
            self.model.eval()
            self.llm_ready = True
            print("✅ Model loaded.")
        except Exception as e:
            self.llm_ready = False
            self.tokenizer = None
            self.model = None
            print("⚠️ Could not load model:", e)

    def _llm_generate(self, prompt: str, max_new_tokens: int = 260, mode: str = "safe") -> str:
        if not getattr(self, "llm_ready", False):
            return ""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            repetition_penalty=1.12,
        )
        if mode == "creative":
            kwargs.update(dict(do_sample=True, top_p=0.92, temperature=0.7, num_beams=1))
        else:
            kwargs.update(dict(do_sample=False, num_beams=4))

        with torch.no_grad():
            out = self.model.generate(**kwargs)

        return (self.tokenizer.decode(out[0], skip_special_tokens=True) or "").strip()

    def _clean_llm_output(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        banned_contains = [
            "rules:",
            "user level:",
            "assistant answer:",
            "do not invent facts",
            "specialised in technology",
            "helpful facts",
        ]

        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            low = s.lower()
            if not s:
                continue
            if any(b in low for b in banned_contains):
                continue
            lines.append(ln)

        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _is_bad(self, out: str) -> bool:
        o = (out or "").strip()
        if len(o) < 25:
            return True
        low = o.lower()
        if low in {"science/technology", "technology", "open ai updates", "chatgpt", "openai"}:
            return True
        if len(set(low.split())) <= 3:
            return True
        return False

    # =========================
    # PREPROCESS / INTENT (IMPROVED)
    # =========================
    def preprocess(self, user_input: str) -> str:
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        self.context["preprocess_meta"] = {
            "cleaned": cleaned,
            "keywords": self._expand_keywords(cleaned),
            "ts": _now_iso(),
        }
        return cleaned.lower()

    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract the main topic from a question."""
        t = (text or "").lower().strip()
        
        # Patterns to extract topic
        patterns = [
            r"(?:what is|what's|whats|explain|tell me about|describe|define)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\?|$)",
            r"(?:how does|how do)\s+(.+?)\s+work",
            r"(?:what are|what're)\s+(.+?)(?:\?|$)",
            r"(?:explain|describe)\s+(.+?)(?:\?|$)",
            r"(?:about|regarding)\s+(.+?)(?:\?|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, t)
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = re.sub(r"[?!.,]+$", "", topic).strip()
                return topic
        
        return None

    def _find_knowledge_match(self, text: str) -> Optional[str]:
        """Find matching knowledge base entry for the query."""
        t = (text or "").lower().strip()
        
        # Extract topic from question
        topic = self._extract_topic(text)
        
        # Direct matches first
        for key in self.tech_knowledge.keys():
            if key in t or (topic and key in topic):
                return key
        
        # Check keyword expansions
        for key, expansions in self.keyword_expansion.items():
            if key in self.tech_knowledge:
                if key in t:
                    return key
                for exp in expansions:
                    if exp in t:
                        return key
        
        return None

    def classify_intent(self, processed_input: str) -> Dict[str, Any]:
        raw = self.context.get("preprocess_meta", {}).get("cleaned", processed_input)
        p = processed_input.strip().lower()

        # Greeting
        if re.search(r"\b(hello|hi|hey|yo|sup|whats up|what's up)\b", p) and len(p.split()) <= 4:
            return {"type": "greeting", "raw": raw}

        # Small talk
        if re.search(r"\b(how are you|how u doing|u good|you good|hru)\b", p) or p in {"ok", "okay", "really"}:
            return {"type": "smalltalk", "raw": raw}

        # Farewell
        if any(x in p for x in ["bye", "quit", "exit"]):
            return {"type": "farewell", "raw": raw}

        wants_sources = bool(re.search(r"\b(link|links|source|sources|url)\b", p))
        wants_resources = bool(re.search(r"\b(learn|resources|course|tutorial|roadmap)\b", p))

        # Check for updates/news FIRST - be more specific
        updates_patterns = [
            r"\b(news|latest|recent|updates?)\b.*\b(about|on|for|from|in)\b",
            r"\bwhat'?s\s+new\b",
            r"\bwhats\s+new\b",
            r"\brecent\s+(news|updates|changes|developments)\b",
            r"\blatest\s+(news|updates|on|about|from)\b",
        ]
        is_updates = any(re.search(pat, p) for pat in updates_patterns)

        # Check for explanation/definition questions (IMPROVED)
        explanation_patterns = [
            r"\b(what is|what's|whats)\b",
            r"\b(explain|describe|define|tell me about)\b",
            r"\b(how does|how do)\b.*\bwork\b",
            r"\b(what are|what're)\b",
            r"\babout\s+\w+",
        ]
        is_explanation = any(re.search(pat, p) for pat in explanation_patterns)

        # Check if we have knowledge about this topic
        knowledge_key = self._find_knowledge_match(p)

        # Decide intent type
        if is_updates and not is_explanation:
            intent_type = "updates"
        elif is_explanation or knowledge_key:
            intent_type = "knowledge" if knowledge_key else "answer"
        else:
            intent_type = "answer"

        return {
            "type": intent_type,
            "raw": raw,
            "wants_sources": wants_sources,
            "wants_resources": wants_resources,
            "knowledge_key": knowledge_key,
            "topic": self._extract_topic(p),
        }

    # =========================
    # RSS / UPDATES
    # =========================
    def _static_rss_sources(self) -> Dict[str, str]:
        return {
            "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            "TechCrunch": "https://techcrunch.com/feed/",
            "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        }

    def _google_news_rss(self, query: str, days: int = 14, hl: str = "en-GB", gl: str = "GB", ceid: str = "GB:en") -> str:
        q = requests.utils.quote(f"{query} when:{days}d")
        return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

    def _fetch_rss(self, url: str, source: str, timeout: int = 12) -> List[Dict[str, Any]]:
        headers = {"User-Agent": "TechBot/1.0"}
        out: List[Dict[str, Any]] = []
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
            for e in feed.entries[:20]:
                title = _clean_text(getattr(e, "title", ""))
                link = _clean_text(getattr(e, "link", ""))
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                summary = _clean_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
                if title:
                    out.append({"source": source, "title": title, "link": link, "published": published, "summary": summary})
        except Exception:
            return []
        return out

    def _fetch_updates(self, query_hint: str, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch updates with better filtering for relevance."""
        key = (query_hint or "general").lower().strip()
        now = datetime.now().timestamp()

        if key in self._news_cache:
            ts, items = self._news_cache[key]
            if (now - ts) < 120:
                return items

        items: List[Dict[str, Any]] = []
        
        # For specific topics, prioritize Google News search
        if query_hint and len(query_hint.strip()) >= 2:
            # Clean up the query for better search
            search_query = query_hint
            # Remove common words
            for word in ["what's", "whats", "new", "on", "about", "latest", "news", "updates"]:
                search_query = re.sub(rf"\b{word}\b", "", search_query, flags=re.IGNORECASE)
            search_query = search_query.strip()
            
            if search_query:
                items.extend(self._fetch_rss(self._google_news_rss(search_query), "Google News"))

        # Only add general sources if we don't have specific results
        if len(items) < 3:
            for src, url in self._static_rss_sources().items():
                items.extend(self._fetch_rss(url, src))

        # Filter for relevance if we have keywords
        if keywords:
            keywords_lower = [k.lower() for k in keywords if len(k) > 2]
            if keywords_lower:
                filtered = []
                for it in items:
                    title_lower = (it.get("title") or "").lower()
                    summary_lower = (it.get("summary") or "").lower()
                    # Check if any keyword matches
                    if any(kw in title_lower or kw in summary_lower for kw in keywords_lower):
                        filtered.append(it)
                # Use filtered if we have results, otherwise fall back
                if filtered:
                    items = filtered

        # Deduplicate
        seen = set()
        deduped = []
        for it in items:
            t = (it.get("title") or "").lower()
            if t and t not in seen:
                seen.add(t)
                deduped.append(it)

        self._news_cache[key] = (now, deduped[:12])
        return deduped[:12]

    # =========================
    # PROMPTS / ANSWERS (IMPROVED)
    # =========================
    def _prompt_general(self, profile: Dict[str, Any], memory: str, question: str) -> str:
        level = profile.get("knowledge_level", "beginner")
        grounding = self._grounding_lines(question)

        return f"""You are a friendly helpful assistant specialized in technology.
Answer naturally. If the user is casual, respond casually too.
Give a helpful, informative answer to the question.

Known facts (only use if relevant):
{grounding if grounding else "(none)"}

User level: {level}
Conversation memory:
{memory if memory else "(none)"}

Question: {question}

Provide a clear, helpful answer:""".strip()

    def _prompt_updates(self, question: str, items: List[Dict[str, Any]], wants_sources: bool) -> str:
        packed = []
        for it in items[:8]:
            line = f"- {it.get('title','')}"
            if it.get("summary"):
                line += f" | {_truncate(it.get('summary',''), 140)}"
            packed.append(line)
        ctx = "\n".join(packed)

        return f"""You are a technology assistant.
Summarize the recent updates based ONLY on the items below. Do not invent details.

User asked: {question}

Items:
{ctx if ctx else "(none)"}

Write:
- 1-2 sentence summary
- 4-6 bullets: what's new + why it matters
{"- Then list sources with links" if wants_sources else ""}

Answer:""".strip()

    def _answer_from_knowledge(self, key: str, profile: Dict[str, Any]) -> str:
        """Return answer from built-in knowledge base."""
        content = self.tech_knowledge.get(key, "")
        if not content:
            return None
        
        level = profile.get("knowledge_level", "beginner")
        
        # For beginners, we might simplify (future enhancement)
        # For now, return the full content
        return content

    def _answer_general(self, profile: Dict[str, Any], question: str, knowledge_key: str = None) -> str:
        """Generate answer - uses knowledge base first, then LLM."""
        
        # Try knowledge base first
        if knowledge_key:
            kb_answer = self._answer_from_knowledge(knowledge_key, profile)
            if kb_answer:
                return kb_answer

        # Fall back to LLM
        user_id = self.context.get("current_user", "default_user")
        mem = self._get_memory_summary(user_id)

        prompt = self._prompt_general(profile, mem, question)
        out = self._llm_generate(prompt, max_new_tokens=280, mode="safe")
        out = self._clean_llm_output(out)

        if self._is_bad(out):
            out2 = self._llm_generate(prompt, max_new_tokens=340, mode="creative")
            out2 = self._clean_llm_output(out2)
            if not self._is_bad(out2):
                return out2

        if self._is_bad(out):
            # Try to find a related topic in knowledge base
            topic = self._extract_topic(question)
            if topic:
                for key in self.tech_knowledge.keys():
                    if key in topic.lower() or topic.lower() in key:
                        return self._answer_from_knowledge(key, profile)
            
            return f"""I'd be happy to help with that! Here are some ways I can assist:

**Try asking me:**
- "What is [technology]?" - e.g., "What is ChatGPT?"
- "Explain [concept]" - e.g., "Explain RAG"
- "What's new in [topic]" - e.g., "What's new in AI"
- "Tell me about [company]" - e.g., "Tell me about OpenAI"

**Topics I know well:**
AI/ML, ChatGPT, OpenAI, NVIDIA, Cloud Computing, Python, JavaScript, React, Databases, Cybersecurity, and more!

What would you like to learn about?"""
        return out

    def _answer_updates(self, question: str, items: List[Dict[str, Any]], wants_sources: bool, keywords: List[str] = None) -> str:
        if not items:
            return "⚠️ I couldn't fetch live updates (RSS blocked/no internet). Try asking a specific question like 'What is ChatGPT?' instead."

        prompt = self._prompt_updates(question, items, wants_sources)
        out = self._llm_generate(prompt, max_new_tokens=360, mode="safe")
        out = self._clean_llm_output(out)

        if self._is_bad(out):
            out2 = self._llm_generate(prompt, max_new_tokens=420, mode="creative")
            out2 = self._clean_llm_output(out2)
            if not self._is_bad(out2):
                return out2

        # fallback: simple list with better formatting
        topic = self._extract_topic(question) or "tech"
        lines = [f"📰 **Recent {topic.title()} News:**\n"]
        for it in items[:5]:
            title = it.get('title', '')
            source = it.get('source', '')
            line = f"• {title}"
            if source:
                line += f" ({source})"
            if wants_sources and it.get("link"):
                line += f"\n  → {it['link']}"
            lines.append(line)
        
        lines.append("\n💡 *For detailed explanations, try: 'What is [topic]?' or 'Explain [concept]'*")
        return "\n".join(lines)

    # =========================
    # MAIN RETRIEVE (IMPROVED)
    # =========================
    def retrieve_info(self, intent: Dict[str, Any], processed_input: str) -> Dict[str, Any]:
        user_id = self.context.get("current_user", "default_user")
        profile = self.get_user_profile(user_id)

        raw = intent.get("raw") or self.context.get("preprocess_meta", {}).get("cleaned", processed_input)
        keywords = self.context.get("preprocess_meta", {}).get("keywords", [])

        t = intent.get("type", "answer")

        if t == "greeting":
            return {"content": [f"👋 Hey! I'm {self.name}. Ask me any tech question - I can explain concepts like ChatGPT, AI, cloud computing, and more!"], "resources": []}

        if t == "smalltalk":
            return {"content": ["I'm good 😄 What tech topic would you like to explore? I can explain concepts, share news, or help you learn!"], "resources": []}

        if t == "farewell":
            return {"content": ["👋 Goodbye! Keep learning!"], "resources": []}

        wants_sources = bool(intent.get("wants_sources", False))
        wants_resources = bool(intent.get("wants_resources", False))
        knowledge_key = intent.get("knowledge_key")

        sections: List[str] = []

        if t == "updates":
            items = self._fetch_updates(query_hint=raw, keywords=keywords)
            sections.append(self._answer_updates(raw, items, wants_sources=wants_sources, keywords=keywords))
        elif t == "knowledge":
            # Use built-in knowledge
            sections.append(self._answer_general(profile, raw, knowledge_key=knowledge_key))
        else:
            sections.append(self._answer_general(profile, raw, knowledge_key=knowledge_key))

        # Optional learning resources
        if wants_resources:
            topic_hint = intent.get("topic") or raw
            res = self._get_personalized_resources(profile, keywords, topic_hint)
            if res:
                sections.append("")
                sections.append("📚 **Learning Resources:**")
                for r in res[:5]:
                    sections.append(f"• {r}")
                    self.track_resource(user_id, topic=topic_hint[:40], resource=r, status="suggested")
                sections.append("\n💡 *Mark done with: `done <keyword>`*")

        self.context["last_topic"] = raw
        return {"content": [s for s in sections if s.strip() or s == ""], "resources": []}

    # =========================
    # generate_response: save turns
    # =========================
    def generate_response(self, retrieved_info: dict) -> str:
        response = super().generate_response(retrieved_info)

        try:
            user_id = self.context.get("current_user", "default_user")
            user_text = ""
            if self.conversation_history:
                user_text = self.conversation_history[-1].get("user", "") or ""
            if user_text and response:
                self._save_turn(user_id, user_text, response)
        except Exception as e:
            print("⚠️ Save turn failed:", e)

        return response
