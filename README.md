---
title: Tennis Match Chat Assistant
emoji: 🎾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: tennis_chat_interface.py
pinned: false
---

# Tennis Match Chat Assistant 🎾

**Free, open-source natural language query system for professional tennis match analysis**

An AI-powered interface that lets you ask natural language questions about professional tennis matches. Get instant, accurate answers about match statistics, player performance, tactical patterns, and detailed point-by-point analysis.

Designed for coaches, analysts, and fans who want to explore 
deep tactical insights from professional matches using Tennis Abstract's 
comprehensive match charting data.

> **What makes this different?** Unlike basic stat displays, this system _understands_ your questions, _computes_ statistics on-demand from point-by-point data, and provides _contextual_ analysis. Ask anything from "How many aces?" to "How did Nadal's serve placement effectiveness evolve when facing break points in Set 3?"


## Features

### Core Capabilities

- **Natural Language Queries**: Ask questions in plain English about any tennis match covered by Tennis Abstract Match Charting Project (15,000+ matches, 2000+ players)
- **Advanced RAG System**: Intelligent retrieval with metadata-enhanced chunking, semantic search, and set-specific filtering
- **Hybrid Query Engine**: Automatically routes queries to optimal computation engine (tree-based aggregation or shot-level analysis)
- **Real-Time Statistics**: Computes statistics on-demand from point-by-point data, not pre-computed aggregates
- **Multi-Tier Complexity**: Automatically scales from simple stats to deep tactical analysis

### Statistical Coverage

- **Serve Analysis**: First serve %, aces, double faults, placement effectiveness (by target, court side, situation)
- **Return Analysis**: Return points won, break point conversion, effectiveness by depth and serve type
- **Shot Analysis**: Winners, errors, directions (crosscourt, down-the-line), shot types (forehand, backhand, volley)
- **Situational Performance**: Break points, game points, deuce points, pressure situations
- **Rally Patterns**: Length distribution, shot sequences, tactical patterns
- **Temporal Analysis**: Set-by-set trends, game-specific analysis, match evolution
- **Comparative Analysis**: Player comparisons, set comparisons, situation comparisons

### Advanced Features

- **Vague Term Resolution**: Ask about "effectiveness", "aggression", "consistency" - system maps to concrete metrics
- **Set-Specific Filtering**: Precisely targets data from specific sets, games, or score situations
- **Shot-Level Queries**: Analyze shot directions, types, and sequences in detail
- **Context-Aware Answers**: Retrieves relevant match context and tactical insights
- **Intelligent Caching**: Instant subsequent loads (first load ~30s, cached <1s)

## Technology Stack

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Frontend** | Gradio 4.44.0 | Mobile-friendly, rapid prototyping, great UX |
| **Backend** | Python 3.x | Rich data science ecosystem |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Local, free, fast, 384-dim vectors |
| **Vector Search** | FAISS | Industry-standard, efficient similarity search |
| **LLM** | Google Gemini 2.5/3.0 Flash | Best quality/cost ratio, large context window |
| **Data Processing** | NumPy, pandas | Fast array operations, data manipulation |
| **Data Source** | Tennis Abstract | Most comprehensive tennis charting data available |

## How It Works

### Architecture Overview

```
User Question → Query Classification → Hybrid Router → Computation Engine
                                            ↓
                               ┌────────────┴────────────┐
                               ↓                         ↓
                        Tree-Based                 Shot-Level
                        Aggregation                Processing
                               ↓                         ↓
                               └────────────┬────────────┘
                                            ↓
                                    RAG Retrieval
                                            ↓
                                    LLM Answer Generation
```

### Processing Pipeline

1. **Query Classification** (100-200ms)
   - Extracts metrics, filters, and grouping dimensions
   - Maps vague terms ("effectiveness") to concrete metrics
   - Detects complexity level (simple → ultra-complex)
   - Determines optimal computation path

2. **Hybrid Routing** (instant)
   - **Tree-Based Route**: For point outcomes, serve stats, basic metrics
   - **Shot-Level Route**: For shot directions, rally patterns, sequence analysis
   - Decision based on query requirements and available optimizations

3. **Computation** (10-500ms depending on complexity)
   - Tree-based: O(n) traversal of point metadata
   - Shot-level: Rally parsing + shot-by-shot analysis
   - Supports filtering, grouping, and multi-dimensional aggregation

4. **RAG Retrieval** (200-500ms)
   - Embeds query (384-dim vector)
   - FAISS similarity search over match chunks
   - Metadata filtering (set/game-specific)
   - Complexity-adaptive chunk selection (5/8/15/18 chunks)

5. **Answer Generation** (2-8s depending on complexity)
   - Assembles context (computed stats + retrieved chunks)
   - Gemini 2.5/3.0 Flash generates contextualized answer
   - Cites specific statistics and tactical insights

### Key Technical Features

- **Vague Term Resolution**: "How effective was the serve?" → `['first_serve_win_pct', 'aces', 'double_faults']`
- **Set-Specific Filtering**: "What happened in Set 3?" → Only retrieves chunks tagged with Set 3
- **Metadata-Enhanced Chunking**: Each chunk tagged with set numbers and game scores
- **Inventory-Driven Filtering**: Discovers shot types from data, no hard-coded lists
- **Smart Caching**: Match data, embeddings, and indexes cached for instant reloads

<!-- Setup and run instructions intentionally omitted in this readme to focus on project overview. -->

## Usage

1. **Search for Players**: Use the dropdown menus to select two players
2. **Browse Matches**: Click "Search Matches" to see their head-to-head history
3. **Load a Match**: Select a match from the list and click "Load Match"
4. **Ask Questions**: Type your question in the chat box

### Example Questions

**Tree-Based Queries** (Fast point-level aggregation - Tier 2: ~4-6s)
- "Who was better at 'stealing' points: winning points when they were down 0–30 or 15–40?"
  - Multi-filter: specific point scores + win rate comparison
  
- "On break points in Sets 4–5, what % of points were 7+ shots, and who won them?"
  - Situation + set range + rally length filter, with win rate grouping
  
- "How did Sinner's return points won % change from Sets 1–2 to Sets 3–5?"
  - Set group comparison with temporal trend analysis

**Shot-Level Queries** (Rally parsing required - Tier 2-3: ~5-8s)
- "What was Sinner's most common 3-shot pattern in rallies?"
  - Requires parsing intermediate shots in rallies (not just serve/return/winning shot)
  - Tree metadata only stores shots 1-3 and the final shot, not sequences
  
- "After Sinner hit an inside-out forehand, did he follow up with a net approach more often than staying at the baseline?"
  - Requires finding all inside-out forehands, then checking the next shot
  - Tree metadata doesn't store shot sequences, only individual shot attributes
  
- "In rallies that went 7+ shots, what was the typical shot direction pattern?"
  - Requires analyzing all shots in long rallies, not just the winning shot
  - Tree metadata only stores serve/return/serve+1/winning shot directions
  - Shot sequences + conditional analysis by court position

**Hybrid Queries** (Computation + Narrative - Tier 3: ~8-12s)
- "Was this match 'Sinner raising level' or 'Medvedev collapsing'? Quantify it."
  - Multi-metric trend analysis + narrative interpretation + quantification
  
- "Describe how break point patterns evolved and which player adapted better"
  - Situational analysis + temporal trends + tactical narrative
  
- "What tactical adjustments did Djokovic make after losing Set 1, and were they effective?"
  - Set-specific comparison + tactical evolution + effectiveness metrics

**Conditional & Multi-Dimensional** (Complex filters - Tier 2-3: ~6-9s)
- "When serving at 30-30 in deuce court, which serve direction was most effective?"
  - Point score + court side + serve direction + effectiveness comparison
  
- "In games where Nadal was broken, what were his serve stats for that game?"
  - Game outcome conditional + filtered aggregation
  
- "How did winner-to-error ratios differ in tight vs comfortable games?"
  - Game pressure classification + ratio calculation + comparison

**Pressure & Momentum Analysis** (Situational + Narrative - Tier 2-3: ~6-10s)
- "Who performed better in the final 3 games of close sets?"
  - Temporal + situational (late-set pressure) + comparative
  
- "After losing 3 consecutive points, what % of 4th points did each player win?"
  - Momentum pattern detection + conditional probability
  
- "What was the longest streak of consecutive service holds, and who broke it?"
  - Sequence detection + game-level analysis + narrative context

**Vague Term Resolution** (System auto-resolves - Tier 1-2: ~3-6s)
- "How effective was the serve under pressure?" 
  - → Analyzes `break_point_saves`, `game_point_conversion`, `serve_win_pct` in pressure situations
  
- "Who was more aggressive in baseline exchanges?"
  - → Compares `winners`, `avg_rally_length`, `forced_errors`, `winner_to_error_ratio` in rallies
  
- "Which player showed better consistency throughout the match?"
  - → Tracks `unforced_errors`, `first_serve_pct`, `point_win_variance` across sets

## System Capabilities

### Intelligent Query Processing

**4-Tier Complexity Detection**:
- **Tier 0 (Simple)**: Direct factual questions → 5 chunks, Gemini 2.5 Flash, minimal thinking
- **Tier 1 (Detailed)**: Context-specific questions → 8 chunks, Gemini 2.5 Flash, low thinking
- **Tier 2 (Moderate)**: Comparative/tactical questions → 15 chunks, Gemini 3.0 Flash, medium thinking
- **Tier 3 (Complex)**: Multi-factor analysis → 18 chunks, Gemini 3.0 Flash, high thinking

**Hybrid Query Routing**:
- **Tree-Based Engine**: Fast aggregation for point outcomes, serve stats (50ms typical)
- **Shot-Level Engine**: Deep analysis for shot directions, rally patterns (500ms typical)
- Automatic routing based on query requirements

**Vague Term Resolution**:
```
"effectiveness" → ['first_serve_win_pct', 'second_serve_win_pct', 'aces']
"aggression" → ['winners', 'unforced_errors', 'avg_rally_length', 'net_approaches']
"pressure performance" → ['break_point_conversion', 'deuce_points_won']
```

### Statistical Capabilities

**70+ Supported Metrics**:
- Point outcomes: `points_won`, `win_percentage`
- Serve: `first_serve_pct`, `aces`, `double_faults`, `serve_points_won`
- Return: `return_points_won`, `break_point_conversion`, `return_winners`
- Situations: `break_points_saved`, `game_point_conversion`, `deuce_points_won`
- Rally: `avg_rally_length`, `short_rally_win_pct`, `long_rally_win_pct`
- Shots: `forehand_winners`, `backhand_errors`, `volley_winners`, `net_points_won`

**Multi-Dimensional Grouping**:
- By set, player, serve direction, court side, situation
- Nested grouping: "Break point conversion by set for each player"
- Conditional filtering: "First serve % when returning on deuce court in Set 3"

**Shot-Level Analysis**:
- Shot types: forehand, backhand, volley, overhead, drop shot
- Directions: crosscourt, down-the-line, inside-out, inside-in
- Depths: shallow, medium, deep, very deep
- Outcomes: winner, forced error, unforced error

### Advanced Analysis Types

- **Temporal Trends**: How statistics evolved across sets/games
- **Conditional Performance**: "When X, what was Y?" queries
- **Comparative Analysis**: Player vs player, set vs set, situation vs situation
- **Tactical Patterns**: Shot sequences, rally patterns, strategic adjustments
- **Momentum Analysis**: Turning points, psychological shifts, match flow

## Performance & Caching

### Caching Strategy

| Data Type | First Load | Cached | Cache Duration |
|-----------|-----------|--------|----------------|
| **Player Names** | ~5-10s (web scrape) | <100ms | 7 days |
| **Match Data** | ~30-60s (scrape + process) | <1s | Indefinite |
| **Embeddings** | ~20-30s (generation) | <1s | Indefinite |
| **Point Tree** | ~5-10s (parsing) | <500ms | Indefinite |

### Query Performance

| Query Type | Example | Typical Response Time |
|------------|---------|----------------------|
| **Simple stats** | "How many aces?" | 2-3 seconds |
| **Tree-based filtered** | "Who won more points at 0-30?" | 4-6 seconds |
| **Set comparisons** | "Return % Sets 1-2 vs 3-5?" | 4-6 seconds |
| **Shot-level** | "Did Sinner target FH or BH more?" | 5-8 seconds |
| **Multi-dimensional** | "Break points in Sets 4-5, 7+ shots?" | 6-9 seconds |
| **Hybrid narrative** | "Sinner raising level or Medvedev collapse?" | 8-12 seconds |

**Performance Breakdown** (typical):
- Query classification: 100-200ms
- Computation: 10-500ms (tree-based) or 200-1000ms (shot-level)
- RAG retrieval: 200-500ms
- LLM generation: 2-8s (varies by complexity and model)

### Resource Usage

**Per Match**:
- Embeddings: ~5MB
- Point tree: ~1MB  
- Chunks: ~500KB
- **Total**: ~6.5MB per match

**Memory Management**: Only loads currently selected match into memory.

<!-- Sharing options intentionally omitted. -->

## Project Structure

```
Tennis/
├── agents/
│   ├── chat_match_questions.py                    # Main query engine (22K+ lines)
│   │   • Query classification
│   │   • Hybrid routing
│   │   • Tree-based aggregation
│   │   • Shot-level processing
│   │   • RAG retrieval
│   │   • LLM integration
│   ├── data_collection_agent.py                   # Web scraper for Tennis Abstract
│   └── tennis_natural_language.py                 # NL report generation
├── tennis_chat_interface.py                       # Gradio web interface
├── ARCHITECTURE.md                                # Technical architecture (this file)
├── README.md                                      # User guide
├── .env                                           # API keys (create this)
└── requirements.txt                               # Python dependencies

Data Files (generated):
├── cached_player_names.json                       # Cached player list
├── {player1}_{player2}_{date}.json                # Match data
├── {player1}_{player2}_{date}_embeddings.pkl      # Embeddings
├── {player1}_{player2}_{date}.index               # FAISS index
└── {player1}_{player2}_{date}_chunks.pkl          # Chunk data
```

## Technical Details

### Data Pipeline

1. **Scraping**: Tennis Abstract match pages → Raw HTML
2. **Parsing**: HTML → Structured JSON (pointlog, metadata, players)
3. **NL Generation**: JSON → Markdown report (~10-50KB per match)
4. **Chunking**: Markdown → 50-200 chunks (500 tokens each)
5. **Embedding**: Chunks → 384-dim vectors (sentence-transformers)
6. **Indexing**: Vectors → FAISS index for similarity search
7. **Point Tree**: Pointlog → Hierarchical metadata structure

### Query Processing

1. **Classification**: Extract metrics, filters, grouping, complexity
2. **Vague Term Resolution**: Map abstract terms to concrete metrics
3. **Routing**: Determine tree-based vs shot-level processing
4. **Computation**: Execute aggregation on point/shot data
5. **Retrieval**: FAISS search + metadata filtering
6. **Context Assembly**: Computed stats + retrieved chunks
7. **LLM Generation**: Gemini generates contextualized answer

### Key Innovations

- **Hybrid Routing**: Automatic selection of optimal computation engine
- **Vague Term Resolution**: Interprets abstract concepts like "effectiveness"
- **Metadata-Enhanced Chunking**: Set/game tagging for precise retrieval
- **Inventory-Driven Filtering**: Discovers filters from data, not hard-coded
- **Complexity-Adaptive Retrieval**: Dynamic chunk count (5/8/15/18)

## Notes

- **Data Source**: Tennis Abstract (https://tennisabstract.com) - 15,000+ matches, 2000+ players
- **Free Embeddings**: Local sentence-transformers model (no API costs)
- **LLM Costs**: Gemini free tier is generous; typical query < $0.001
- **Accuracy**: Statistics computed directly from point-by-point data
- **Extensibility**: Add new metrics, filters, or groupings easily
- **Platform**: Fully tested on Windows 10/11
- **Mobile**: Gradio interface is responsive and mobile-friendly

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete technical architecture and implementation details
- **README.md** (this file): User guide and quick reference
- Inline code documentation: Extensive comments throughout `chat_match_questions.py`

<!-- Troubleshooting intentionally omitted. -->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

This project is for educational and research purposes.

## About the Author

Developed by **Rami Zheman**, data and AI consultant passionate about sports analytics (in addition to his day job in IT government contracting!).

This project represents 6+ months of development, iterating on query understanding, statistical computation, and retrieval optimization. The 22,000+ line query engine is the result of handling edge cases from hundreds of real user questions.

Connect on [LinkedIn](https://linkedin.com/in/ramizheman) or [GitHub](https://github.com/ramizheman).

### Development Stats

- **Lines of Code**: ~25,000+ (main engine: 22,831 lines)
- **Supported Metrics**: 70+
- **Complexity Tiers**: 4 (auto-detected)
- **Match Coverage**: 15,000+ matches from Tennis Abstract
- **Development Time**: 6+ months
- **Test Matches**: 100+ matches tested across ATP/WTA

## Acknowledgments

- Tennis Abstract for providing detailed match charting data
- The tennis analytics community for their insights
- All the contributors to the open-source libraries used in this project

---

