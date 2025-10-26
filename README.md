# Tennis Match Chat Assistant 🎾

An AI-powered chat interface that lets you ask natural language questions about professional tennis matches. Get instant answers about match statistics, player performance, and detailed point-by-point analysis.

## Features

- **Natural Language Queries**: Ask questions in plain English about any tennis match
- **Advanced RAG System**: Intelligent retrieval with metadata-enhanced chunking and semantic search
- **Player Search**: Search and filter through 2000+ professional players (ATP & WTA)
- **Match History**: Browse complete head-to-head match history between any two players
- **Multi-Tier Analysis**: Automatically scales complexity from simple stats to deep tactical analysis
- **Per-Set Analytics**: Count and analyze statistics on a per-set or per-game basis
- **Comprehensive Coverage**:
  - Match scores and set-by-set results
  - Serve statistics (aces, double faults, first serve %, placement effectiveness)
  - Break point conversions and save rates
  - Shot directions, types, and outcomes
  - Rally patterns and point-by-point narratives
  - Tactical evolution throughout the match
  - Mental and momentum shifts

## Technology Stack

- **Frontend**: Gradio web interface (mobile-friendly)
- **Backend**: Python
- **Embeddings**: Sentence Transformers (local, free)
- **LLM**: Google Gemini 2.5 Flash (default, configurable for OpenAI/Claude)
- **Vector Search**: FAISS
- **Data Source**: Tennis Abstract match charting data

## Setup

### Prerequisites

- Python 3.12+
- Git
- A Google Gemini 2.5 API key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ramizheman/Tennis.git
cd Tennis
```

2. **Create a virtual environment**
```bash
python -m venv tennis-venv
# On Windows:
tennis-venv\Scripts\activate
# On Mac/Linux:
source tennis-venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up your API key**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

### Running the Application

```bash
python tennis_chat_interface.py
```

The interface will be available at: `http://localhost:7860`

## Usage

1. **Search for Players**: Use the dropdown menus to select two players
2. **Browse Matches**: Click "Search Matches" to see their head-to-head history
3. **Load a Match**: Select a match from the list and click "Load Match"
4. **Ask Questions**: Type your question in the chat box

### Example Questions

**Simple Stats:**
- "What was the final score?"
- "How many aces did each player hit?"
- "How many forehand winners did each player hit?"
- "How many double faults did each player have?"

**Set-Specific Analysis:**
- "What happened in Set 3?"
- "Compare Player 1's unforced errors in Set 1 versus Set 5"
- "How did Player 2's serve effectiveness change across sets?"

**Tactical Comparisons:**
- "When Player 1 served to the T versus wide, which was more effective?"
- "What were Player 2's break point conversion statistics?"
- "Who controlled rallies from the baseline more effectively?"

**Advanced Analysis:**
- "How did Player 1's first serve percentage change when facing break points?"
- "Tell the parallel journey of both players through the match - how did each player's mental and tactical state evolve?"
- "Provide a summary of the match"

## System Capabilities

### Intelligent Retrieval
- **4-Tier Complexity Detection**: Automatically determines optimal number of chunks (5, 8, 15, or 18) based on query complexity
- **Set/Game Filtering**: Precisely retrieves data for specific sets or games
- **Metadata-Enhanced Chunking**: Each chunk tagged with set numbers and game scores for accurate filtering
- **Point-by-Point Prioritization**: Complex tactical questions automatically retrieve narrative data

### Advanced Question Types
- **Simple Statistics**: Direct answers from aggregate tables (aces, winners, double faults)
- **Set-Specific Queries**: Per-set breakdowns counted from point-by-point data
- **Conditional Analysis**: "When X happened, how did Y respond?" questions
- **Tactical Evolution**: Track strategy changes throughout the match
- **Momentum Analysis**: Identify turning points and psychological shifts

## Caching & Performance

The system implements intelligent caching to improve load times:

- **Player Names**: Cached for 7 days
- **Match Data**: Cached after first load (instant subsequent loads)
- **Embeddings**: Generated once per match, stored locally (never regenerated)

First-time match loads take ~30-60 seconds (scraping + embedding generation).
Subsequent loads of the same match are nearly instant!

## Sharing with Others

### Option 1: ngrok (Recommended)

1. Download ngrok: https://ngrok.com/download
2. Run the tennis interface in one terminal
3. In another terminal:
```bash
ngrok http 7860
```
4. Share the generated `https://xxx.ngrok-free.app` URL

The ngrok URL works on mobile, tablet, and desktop!

### Option 2: Local Network

Share your local IP address with users on the same WiFi network:
```
http://YOUR_IP_ADDRESS:7860
```

## Project Structure

```
Tennis/
├── agents/
│   ├── chat_agent_embedding_qa_local.py    # Main chat agent
│   ├── data_collection_agent.py            # Web scraper
│   └── ...
├── tennis_chat_interface.py                # Gradio interface
├── run_data_collection_agent.py            # Data collection script
├── .env                                    # API keys (create this)
└── requirements.txt                        # Python dependencies
```

## Notes

- **Data Source**: All match data is scraped from Tennis Abstract (https://tennisabstract.com)
- **Free to Use**: Local embeddings model means no embedding API costs
- **LLM Costs**: Gemini has a generous free tier; typical queries cost fractions of a cent
- **Windows Compatible**: Fully tested on Windows 10/11
- **Mobile Friendly**: The Gradio interface is responsive and works great on phones

## Troubleshooting

### Match loading fails
- Check your internet connection (needs to scrape Tennis Abstract)
- Verify the match exists on Tennis Abstract
- Check the terminal for error messages

### API errors
- Verify your `.env` file has the correct API key
- Check your API quota hasn't been exceeded
- Ensure your API key has the necessary permissions

### Slow performance
- First load of a match is always slower (scraping + embedding)
- Subsequent loads use cached data (much faster)
- Consider using a machine with more RAM for faster embedding generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and research purposes.

## Acknowledgments

- Tennis Abstract for providing detailed match charting data
- The tennis analytics community for their insights
- All the contributors to the open-source libraries used in this project

---

**Built with ❤️ for tennis fans and data enthusiasts**
