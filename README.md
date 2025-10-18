# Tennis Match Chat Assistant 🎾

An AI-powered chat interface that lets you ask natural language questions about professional tennis matches. Get instant answers about match statistics, player performance, and detailed point-by-point analysis.

## Features

- **Natural Language Queries**: Ask questions in plain English about any tennis match
- **Comprehensive Match Data**: Access detailed statistics from Tennis Abstract's match charting data
- **Player Search**: Search and filter through 2000+ professional players (ATP & WTA)
- **Match History**: Browse complete head-to-head match history between any two players
- **Instant Analysis**: Get answers about:
  - Match scores and results
  - Serve statistics (aces, double faults, first serve %)
  - Break point conversions
  - Shot directions and outcomes
  - Rally lengths and outcomes
  - Point-by-point narratives
  - And much more...

## Technology Stack

- **Frontend**: Gradio web interface (mobile-friendly)
- **Backend**: Python
- **Embeddings**: Sentence Transformers (local, free)
- **LLM**: Google Gemini (configurable for OpenAI/Claude)
- **Vector Search**: FAISS
- **Data Source**: Tennis Abstract match charting data

## Setup

### Prerequisites

- Python 3.12+
- Git
- A Google Gemini API key (free tier available)

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
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key at: https://makersuite.google.com/app/apikey

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

- "What was the final score?"
- "How many aces did each player hit?"
- "What were the break point statistics?"
- "Who won more points at the net?"
- "What was the longest rally?"
- "Give me a comprehensive analysis of the match"

## Caching & Performance

The system implements intelligent caching to improve load times:

- **Player Names**: Cached for 7 days
- **Match Data**: Cached after first load (instant subsequent loads)
- **Embeddings**: Pre-computed and cached per match

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
