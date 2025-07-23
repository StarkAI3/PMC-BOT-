# PMC Chatbot

A terminal-based chatbot for the Pune Municipal Corporation (PMC) that supports both English and Marathi. The bot uses Google Gemini for language understanding and Pinecone for semantic search over PMC data.

## Features
- Multilingual support: English and Marathi
- Real-time semantic search using Pinecone
- Contextual answers using Gemini LLM
- Data sourced from PMC documents and links

## Project Structure
- `chatbot/terminal_chatbot.py`: Main chatbot script
- `data/`: Contains PMC data in JSONL format
- `scripts/`: Data extraction, normalization, embedding, and upsert scripts
- `requirements.txt`: Python dependencies

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/StarkAI3/PMC-BOT-.git
   cd PMC-BOT-
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   Create a `.env` file with the following keys:
   ```env
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX=pmc-bot-index
   GEMINI_API_KEY=your-gemini-api-key
   ```

## Usage
Run the chatbot in your terminal:
```bash
python chatbot/terminal_chatbot.py
```
Type your queries in English or Marathi. Type `exit` or `quit` to stop the bot.

## Notes
- Make sure your Pinecone and Gemini API keys are valid and have sufficient quota.
- Data files in `data/` can be updated as needed.

## License
This project is for educational and research purposes. 