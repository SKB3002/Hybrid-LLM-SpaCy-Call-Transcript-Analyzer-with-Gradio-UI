import gradio as gr
import pandas as pd
import os, json, requests, re
import spacy
from textblob import TextBlob

# --- Load Groq API Key ---
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")

# --- Load SpaCy model ---
nlp = spacy.load("en_core_web_sm")

# --- Function to summarize transcript using Groq LLM ---
def summarize_with_groq(transcript: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"
    Summarize this customer service call in 1–2 sentences.
    Transcript: {transcript}
    Strictly return only the summary and nothing else"
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=data)
    text = response.json()["choices"][0]["message"]["content"].strip()
    return text

# --- Function to get sentiment ---
def get_sentiment(transcript: str):
    blob = TextBlob(transcript)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- Function to extract Order ID using regex ---
def extract_order_id(transcript: str):
    # Match #12345 or ORD-12345 (case-insensitive)
    pattern = r"(?:#|ORD-?|ID?)\d+"
    match = re.search(pattern, transcript, re.IGNORECASE)
    return match.group(0) if match else None

# --- Function to extract entities: customer name, order ID, product ---
def extract_entities(transcript: str):
    doc = nlp(transcript)
    customer_name = None
    order_id = extract_order_id(transcript)
    product = None

    # Extract customer name
    
    customer_lines = [line for line in transcript.split("\n") if line.strip().lower().startswith("customer:")]
    
    for line in customer_lines:
        # Look for phrases like "this is John Miller" or "my name is John Miller"
        match = re.search(r"(?:this is|my name is|I am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", line, re.IGNORECASE)
        if match:
            customer_name = match.group(1)
            break  # stop after the first match

    # Simple keyword check for product categories
    for prod in ["mobile", "laptop", "refrigerator", "washing machine", "TV"]:
        if prod.lower() in transcript.lower():
            product = prod.capitalize()

    return customer_name, order_id, product

# --- Main function to analyze transcript and save to CSV ---
def analyze_and_save(transcript):
    if not transcript.strip():
        return "⚠️ Please enter a transcript.", "", "", "", "", ""

    summary = summarize_with_groq(transcript)
    sentiment = get_sentiment(transcript)
    customer_name, order_id, product = extract_entities(transcript)

    # Save to CSV
    df = pd.DataFrame([[transcript, summary, sentiment, customer_name, order_id, product]],
                      columns=["Transcript", "Summary", "Sentiment", "Customer Name", "Order ID", "Product"])
    if not os.path.exists("call_analysis.csv"):
        df.to_csv("call_analysis.csv", index=False)
    else:
        df.to_csv("call_analysis.csv", mode="a", header=False, index=False)

    return transcript, summary, sentiment, customer_name, order_id, product

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green")) as demo:
    gr.Markdown(
        "<h1 style='text-align:center; color:#2563eb;'>📞 Call Transcript Analyzer (Hybrid LLM + SpaCy)</h1>"
        "<p style='text-align:center; color:#16a34a;'>LLM-powered summaries with SpaCy + regex-based entity extraction!</p>"
    )

    with gr.Row():
        transcript_in = gr.Textbox(
            label="Enter Call Transcript",
            placeholder="e.g. Hi, I tried to book a slot but payment failed...",
            lines=4
        )

    analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary")

    with gr.Row():
        transcript_out = gr.Textbox(label="Original Transcript", interactive=False)
        summary_out = gr.Textbox(label="Summary", interactive=False)
        sentiment_out = gr.Textbox(label="Sentiment", interactive=False)
        customer_out = gr.Textbox(label="Customer Name", interactive=False)
        order_out = gr.Textbox(label="Order ID", interactive=False)
        product_out = gr.Textbox(label="Product", interactive=False)
    
    analyze_btn.click(
        analyze_and_save,
        inputs=[transcript_in],
        outputs=[transcript_out, summary_out, sentiment_out, customer_out, order_out, product_out]
    )

demo.launch()
