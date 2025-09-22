# 📞 Call Transcript Analyzer (Hybrid LLM + SpaCy)

**Hybrid AI solution for analyzing customer call transcripts:**  
- Summarizes calls using **Groq LLM**  
- Detects **sentiment** using **TextBlob**  
- Extracts **customer names, order IDs, and product mentions** using **SpaCy + Regex**  
- Interactive **Gradio UI**  
- Saves structured insights to **CSV**

---

## 🚀 Features
- **LLM-Powered Summary:** Generate concise 1–2 sentence summaries of call transcripts.
- **Sentiment Analysis:** Classify transcript as Positive / Neutral / Negative.
- **Named Entity Recognition (NER):** Extract customer name, order ID, and product.
- **CSV Logging:** Store all results for analytics or reporting.
- **Interactive UI:** Easy-to-use dashboard with Gradio.

---
## 💻 Installation

bash
### Clone repo
git clone https://github.com/<your-username>/call-transcript-analyzer.git
cd call-transcript-analyzer

### Install dependencies
pip install -r requirements.txt

## 📝 Usage
python app.py
Open the Gradio UI in your browser.
Paste your call transcript.
Click “Analyze Transcript”.
View summary, sentiment, customer name, order ID, and product.
Results automatically saved to call_analysis.csv.

## 📂 Sample Transcript
Agent: Hello, this is Sarah from TechCare support. May I know your name, please?
Customer: Hi Sarah, this is John Miller. I had placed an order for a mobile phone, order ID #45892.

