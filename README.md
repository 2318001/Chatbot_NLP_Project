# Chatbot_NLP_Project



# TechBot - Advanced NLP Tech Assistant

An intelligent chatbot for tech news and learning guidance, built for NLP assessment.

## Features

 **Advanced NLP Processing**
- Multi-intent detection (news, explanations, resources, comparisons)
- Keyword expansion and semantic understanding
- Context-aware responses

**Personalized User Experience**
- User profiling (knowledge level, interests, goals)
- Level-appropriate content filtering
- Personalized resource recommendations

 **Local Operation**
- Runs entirely on your machine
- SQLite database for user data
- Optional local LLM (DistilGPT2)

 **Educational Focus**
- Tech news summarization
- Concept explanations
- Learning path recommendations
- Project suggestions







/// step to run

python -m venv venv
venv\Scripts\activate


pip install -r requirements.txt


python run.py



//for huggingface download

$env:HF_HOME="D:\hf_cache"
$env:TRANSFORMERS_CACHE="D:\hf_cache"
python run.py
