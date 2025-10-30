# Tennis Match Chat Assistant 🎾

**Free, open-source Q&A AI for analyzing pro tennis matches using real match data.**

An AI-powered interface that lets you ask natural language questions about professional tennis matches. Get instant answers about match statistics, player performance, and detailed point-by-point analysis.

Designed for coaches, analysts, and fans who want to explore tactical insights from professional matches — without paying for expensive data feeds.

👉 [View on GitHub](https://github.com/ramizheman/Tennis)

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
- **LLM**: Google Gemini 2.5 Flash (default, configurable as needed)
- **Vector Search**: FAISS
- **Data Source**: Tennis Abstract match charting data

## How It Works

The system combines natural language understanding with a retrieval-augmented generation (RAG) pipeline. Each match is transformed into structured and narrative representations that the model can reason over.

When you ask a question, it retrieves the most relevant match segments, analyzes them, and builds a contextualized answer that blends quantitative stats with tactical insights.

The system automatically adjusts how much contextual data it retrieves and analyzes depending on query complexity — from quick stats to in-depth tactical breakdowns. It dynamically scales from simple queries ("How many aces?") to complex ones ("How did Nadal adjust his return depth in Set 3 vs Set 1?") by automatically retrieving the right level of match detail.

### Key Features
- **Intelligent Retrieval**: Automatically finds the most relevant match data for any question type
- **Multi-Scale Analysis**: Handles everything from basic stats to complex tactical comparisons
- **Set-Specific Filtering**: Precisely targets data from specific sets or game ranges
- **Smart Caching**: Reuses processed data for instant subsequent loads
- **Dependent Search**: Player 2 dropdown automatically shows only real opponents of Player 1

<!-- Setup and run instructions intentionally omitted in this readme to focus on project overview. -->

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
- "How did Player 2's tactics change in Set 1 versus Set 3?"
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

<!-- Sharing options intentionally omitted. -->

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

<!-- Troubleshooting intentionally omitted. -->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

This project is for educational and research purposes.

## About the Author

Developed by **Rami Zheman**, data and AI consultant passionate about sports analytics (in addition to his day job in IT government contracting!)

Connect on [LinkedIn](https://linkedin.com/in/ramizheman) or [GitHub](https://github.com/ramizheman).

## Acknowledgments

- Tennis Abstract for providing detailed match charting data
- The tennis analytics community for their insights
- All the contributors to the open-source libraries used in this project

---

