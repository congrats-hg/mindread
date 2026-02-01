# CLAUDE.md

> Instructions for AI assistants working on this repository.

---

## Project Goal

Build a **state-of-the-art Dialogue State Tracking model** on the DSTC2 benchmark that:
1. Achieves >75% joint goal accuracy (current SOTA: ~72%)
2. Results in a publishable research paper
3. Serves as a standout ML portfolio piece

---

## Research Context

**Benchmark:** DSTC2 (Dialog State Tracking Challenge 2)
- Domain: Restaurant information
- Task: Predict user's goal (food type, area, price range) at each dialogue turn
- Metric: Joint Goal Accuracy

**Research Angle:** Context-aware intent classification — how dialogue history and system architecture influence state tracking accuracy and user interaction patterns.

**Connection to H-AI Research:** This work will analyze how model failures correlate with user adaptation behaviors, bridging NLP engineering with Human-AI Interaction research.

---

## Technical Direction

- Use pretrained language models (BERT, RoBERTa) as the backbone
- Implement hierarchical context encoding for multi-turn dialogue
- Compare slot classification vs. span extraction approaches
- Conduct thorough error analysis to identify failure patterns

---

## Portfolio Standards

**Code Quality:**
- Clean, modular, well-documented code
- Type hints and docstrings throughout
- Comprehensive unit tests

**Reproducibility:**
- Single-command training and evaluation
- Configuration-driven experiments (Hydra or similar)
- Logged experiments with W&B or MLflow

**Documentation:**
- Professional README with results table, architecture diagram, and quick start
- Clear API documentation
- Jupyter notebooks for data exploration and result visualization

**Presentation:**
- Interactive demo (Gradio or Streamlit)
- Visualizations of attention patterns and error distributions
- Comparison tables against published baselines

---

## Key Deliverables

1. **Model:** BERT-based DST model beating current SOTA
2. **Paper:** LaTeX draft following ACL/EMNLP format
3. **Demo:** Interactive web interface for dialogue state tracking
4. **Analysis:** Error analysis connecting to H-AI research themes

---

## Reference Resources

- DSTC2 Data: https://github.com/matthen/dstc
- Key Papers: Neural Belief Tracker, SUMBT, TRADE, SOM-DST
- Research Interest: https://modulabs.co.kr/labs/655 (AI-Art Interaction Lab)

---

## When Building This Repository

- Prioritize clarity over cleverness
- Make it easy for others to reproduce results
- Write code as if it will be reviewed by a hiring manager
- Document decisions and experiment results thoroughly
- Keep the research narrative clear: problem → approach → results → analysis
