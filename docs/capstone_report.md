# Enterprise AI Document Query Agent - Capstone Report

## 1. System Setup

### Prerequisites
- Python 3.11
- OpenAI API key (`.env`)
- Documents in `./data` (PDF/TXT/CSV/Excel)

### Installation
```bash
git clone https://github.com/SprawaDeals/enterprise-ai-document-agent.git
cd enterprise-ai-document-agent
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run app
