# Mechanistic Interpretability for Time Series Foundation Models
## Complete Literature Review & Research Repository

**Last Updated:** March 11, 2026

---

## 📋 Overview

This repository contains a comprehensive literature review and research resources for **Mechanistic Interpretability in Time Series Foundation Models (TSFMs)**. It's designed to serve as a complete reference library for researchers, practitioners, and students interested in this emerging field.

### What's Included

1. **Comprehensive Literature Review** - Complete survey of the field
2. **BibTeX References** - All papers in citation-ready format
3. **Quick Start Guide** - Practical implementation roadmap
4. **Research Roadmap** - Future directions and open problems
5. **This README** - Navigation and usage guide

---

## 📚 Document Guide

### 1. Main Literature Review
**File:** `TSFM_Mechanistic_Interpretability_Literature_Review.md`

**Contents:**
- Introduction to TSFMs and interpretability challenges
- Complete survey of time series foundation models
- Core interpretability papers (TimeSAE, mechanistic TSC)
- Interpretability methodologies (SAEs, activation patching, probing, etc.)
- Implementation resources and code repositories
- Architectural paradigms and design patterns
- Benchmarks and datasets
- Open problems and future directions
- Appendices with quick references

**When to use:**
- Getting comprehensive overview of the field
- Looking up specific papers or models
- Understanding different interpretability methods
- Identifying research gaps

**Reading time:** 3-4 hours (full document), 30 min (specific sections)

---

### 2. BibTeX References
**File:** `references.bib`

**Contents:**
- All papers cited in the literature review
- Organized by category (Foundation Models, Surveys, Interpretability, etc.)
- Includes URLs, GitHub links, and notes
- Ready to use with LaTeX/BibTeX

**When to use:**
- Writing papers or reports
- Creating bibliographies
- Managing citations
- Finding original sources

**Usage:**
```latex
\cite{timesae2026}  % TimeSAE paper
\cite{timesfm2024}  % TimesFM paper
\cite{fm4ts_survey2024}  % Foundation Models Survey
```

---

### 3. Quick Start Guide
**File:** `Quick_Start_Guide.md`

**Contents:**
- Curated reading paths for different audiences
- Implementation checklists
- Code templates (Python)
- Learning resources
- Common pitfalls and solutions
- Weekly research update template

**When to use:**
- Just getting started with TSFMs
- Implementing your first model
- Adding interpretability to existing system
- Teaching or mentoring others

**Target audiences:**
- **Beginners:** Foundation model basics path
- **Researchers:** Interpretability focus path
- **Practitioners:** Production implementation path

---

### 4. Research Roadmap
**File:** `Research_Roadmap.md`

**Contents:**
- High-priority research directions
- Methodological innovations
- Novel application domains
- Benchmark and evaluation frameworks
- Tooling and infrastructure ideas
- PhD thesis ideas
- Grant proposal outlines
- Collaboration opportunities

**When to use:**
- Planning research projects
- Identifying thesis topics
- Writing grant proposals
- Finding collaboration opportunities
- Understanding field trajectory

**Best for:** Graduate students, researchers, grant writers

---

## 🗺️ Navigation Guide

### I'm new to TSFMs. Where do I start?

1. **Read:** `Quick_Start_Guide.md` → "Path 1: Foundation Model Basics"
2. **Skim:** `TSFM_Mechanistic_Interpretability_Literature_Review.md` → Sections 1-3
3. **Try:** Clone [TimesFM](https://github.com/google-research/timesfm) and run example
4. **Return:** To literature review for deeper dive

---

### I want to understand interpretability methods. Where do I go?

1. **Read:** `TSFM_Mechanistic_Interpretability_Literature_Review.md` → Section 4 (Mechanistic Interpretability)
2. **Read:** `Quick_Start_Guide.md` → "Path 2: Interpretability Focus"
3. **Implement:** Code templates in Quick Start Guide
4. **Read:** TimeSAE paper in detail

---

### I'm looking for specific papers or models.

**Use the Literature Review appendices:**
- **Appendix A:** Model comparison table
- **Appendix B:** Interpretability methods comparison
- **Appendix C:** Research timeline
- **Appendix D:** Recommended reading order

**Or search the BibTeX file:**
- Open `references.bib`
- Search by author, title, or keyword
- Follow URLs to papers

---

### I want to contribute to research. Where do I start?

1. **Read:** `Research_Roadmap.md` → High-priority directions
2. **Choose:** 1-2 directions that match your interests/skills
3. **Read:** Relevant papers from literature review
4. **Plan:** Use "Getting Started with Research" section
5. **Collaborate:** Find collaborators via suggested opportunities

---

### I need to implement interpretability in production.

1. **Read:** `Quick_Start_Guide.md` → "Path 3: Production Implementation"
2. **Use:** Implementation checklist in Quick Start Guide
3. **Reference:** Code templates for activation extraction, SAEs, patching
4. **Check:** Common pitfalls section
5. **Deploy:** Using guidance in Quick Start Guide

---

## 📊 Quick Reference Tables

### Foundation Models Covered

| Model | Organization | Year | Type | Zero-Shot | Code |
|-------|-------------|------|------|-----------|------|
| TimeGPT-1 | Nixtla | 2023 | Forecasting | ✓ | API |
| TimesFM | Google | 2024 | Forecasting | ✓ | ✓ |
| Chronos | Amazon | 2024 | Probabilistic | ✓ | ✓ |
| MOMENT | Academic | 2024 | Multi-task | ✓ | ✓ |
| Time-LLM | Academic | 2024 | LLM-based | ✓ | ✓ |
| UniTS | Harvard | 2024 | Unified | ✓ | ✓ |
| Lag-Llama | Academic | 2024 | Probabilistic | ✓ | ✓ |

### Interpretability Methods

| Method | Complexity | Causal? | Best For |
|--------|-----------|---------|----------|
| Sparse Autoencoders | High | With steering | Feature discovery |
| Activation Patching | Medium | ✓ | Causal importance |
| Attention Saliency | Low | ✗ | Quick visualization |
| Probing Classifiers | Low | Partial | Representation testing |
| Integrated Gradients | Medium | Partial | Input attribution |

### Key Papers to Read

**Must-read (start here):**
1. TimeSAE (2026) - First mechanistic interpretability for TSFMs
2. Foundation Models Survey (2024) - Comprehensive overview
3. TimesFM or Chronos paper - Understand architecture

**Should-read (for researchers):**
4. Mechanistic Interpretability for TSC (2025)
5. Towards Monosemanticity (Anthropic)
6. Toy Models of Superposition (Anthropic)

**Nice-to-have (for specialists):**
7. Time-LLM paper
8. UniTS paper
9. Sparse Autoencoders Find Highly Interpretable Features
10. Related architecture papers (PatchTST, etc.)

---

## 🔍 Search Tips

### Finding Specific Topics in Literature Review

**Search for keywords:**
- "Sparse Autoencoder" or "SAE"
- "Activation Patching"
- "Chronos" or "TimesFM" or "TimeGPT"
- "Probing" or "Attribution"
- "Benchmark" or "Dataset"

**Navigate by section numbers:**
- Section 3: Core foundation models
- Section 4: Mechanistic interpretability papers
- Section 5: Interpretability methods
- Section 6: Implementation resources
- Section 9: Open problems

### Finding Papers in BibTeX

**Search patterns:**
```bibtex
# By model name
@article{timesae2026
@article{timesfm2024
@article{chronos2024

# By method
@article{sae_monosemantic2023
@article{shap2017

# By type
@inproceedings{  # Conference papers
@article{        # Journal/preprint papers
@misc{           # Websites, repos
```

---

## 📈 Keeping Up to Date

### How to Stay Current

1. **ArXiv Alerts:**
   - Subscribe to cs.LG, cs.AI tags
   - Keywords: "time series", "foundation model", "interpretability"

2. **Conference Proceedings:**
   - NeurIPS, ICML, ICLR, KDD (major ML conferences)
   - TSALM workshop (specialized)

3. **Key Researchers:**
   - Follow on Twitter/GitHub/Google Scholar
   - Set up citation alerts

4. **Repositories:**
   - Watch GitHub repos (TimesFM, Chronos, etc.)
   - Check for new releases and issues

5. **Community:**
   - r/MachineLearning
   - Papers With Code - Time Series tag
   - Hugging Face - Time series models

### Updating This Repository

**Update frequency:**
- Major papers: Within 1 month of publication
- Minor papers: Quarterly
- Benchmarks/datasets: As needed
- Tools/code: Continuous

**Contribution guidelines:**
- Submit pull requests for new papers
- Follow existing format
- Include all metadata (arXiv ID, GitHub, etc.)
- Update summary tables

---

## 🎓 Learning Path Recommendations

### For Undergraduate Students
**Timeline: 1-2 months**

**Week 1-2: Foundations**
- Read: Literature Review Sections 1-2
- Complete: Quick Start Path 1
- Implement: Run TimesFM demo

**Week 3-4: Deep Dive**
- Read: 2-3 foundation model papers
- Try: Multiple models (Chronos, UniTS)
- Understand: Architecture differences

**Week 5-8: Interpretability**
- Read: Literature Review Section 4-5
- Complete: Quick Start Path 2 (simplified)
- Implement: Basic attention visualization

**Outcome:** Understand what TSFMs are, can use them, basic interpretability

---

### For Graduate Students
**Timeline: 3-6 months**

**Month 1-2: Comprehensive Understanding**
- Read: Full Literature Review
- Complete: Quick Start Path 2
- Implement: SAE on simple TSFM
- Reproduce: TimeSAE key results

**Month 3-4: Research Exploration**
- Read: Research Roadmap
- Choose: 1-2 research directions
- Design: Initial experiments
- Read: 10+ additional papers

**Month 5-6: Research Execution**
- Implement: Novel method or application
- Validate: On benchmarks
- Write: Initial paper draft
- Present: At reading group or workshop

**Outcome:** Ready to publish, clear thesis direction

---

### For Industry Practitioners
**Timeline: 2-4 weeks**

**Week 1: Rapid Onboarding**
- Skim: Literature Review (key sections only)
- Complete: Quick Start Path 3
- Choose: Production model (TimesFM/Chronos)

**Week 2: Implementation**
- Set up: Model serving infrastructure
- Integrate: With existing data pipeline
- Test: Zero-shot performance
- Evaluate: Business metrics

**Week 3: Interpretability (Optional)**
- Add: Basic interpretability (attention viz)
- Implement: Production monitoring
- Document: Model behavior
- Present: To stakeholders

**Week 4: Deployment**
- Deploy: To production
- Monitor: Performance and drift
- Iterate: Based on feedback
- Scale: As needed

**Outcome:** Working TSFM in production with basic interpretability

---

## 🤝 Contributing

### How to Contribute

**Add new papers:**
1. Find paper not in literature review
2. Add to appropriate section
3. Update BibTeX file
4. Update summary tables
5. Submit pull request

**Improve documentation:**
1. Fix typos or errors
2. Add clarifications
3. Improve code examples
4. Update broken links
5. Add new sections

**Share implementations:**
1. Add code examples
2. Create tutorials
3. Share datasets
4. Provide benchmarks

### Contribution Guidelines

**Quality standards:**
- Verify all citations
- Test all code examples
- Check all URLs
- Follow existing format
- Be comprehensive

**Review process:**
1. Submit pull request
2. Maintainer review
3. Feedback and revisions
4. Merge to main

---

## 📞 Getting Help

### Documentation Issues
- Check: This README first
- Search: Literature Review index
- Try: Quick Start Guide troubleshooting

### Technical Questions
- Stack Overflow: `[time-series]` `[transformer]` tags
- GitHub Issues: Model-specific repositories
- Reddit: r/MachineLearning

### Research Collaboration
- Email: Paper authors (polite, specific questions)
- Conferences: NeurIPS, ICML, ICLR, KDD
- Workshops: TSALM, FM4TS
- Online: ResearchGate, Semantic Scholar

---

## 📜 Citation

If you use this literature review in your research, please cite:

```bibtex
@misc{tsfm_interp_lit_review_2026,
  title={Mechanistic Interpretability for Time Series Foundation Models: A Comprehensive Literature Review},
  author={Your Name},
  year={2026},
  month={March},
  howpublished={\\url{https://github.com/yourusername/tsfm-interpretability-review}},
  note={Complete literature review for NotebookLLM library}
}
```

---

## 📄 License

This literature review is provided for educational and research purposes.

**Papers cited:** Each paper has its own license; please respect authors' rights.

**Code examples:** Provided under MIT License unless otherwise specified.

**This documentation:** CC BY 4.0 - share and adapt with attribution.

---

## 🙏 Acknowledgments

**Key contributors to the field:**
- Nixtla (TimeGPT)
- Google Research (TimesFM)
- Amazon Science (Chronos)
- Anonymous authors (TimeSAE)
- Harvard Zitnik Lab (UniTS)
- Anthropic (Interpretability research)

**Resources used:**
- arXiv
- OpenReview
- Papers With Code
- Google Scholar
- Semantic Scholar

**Inspirations:**
- Anthropic's Transformer Circuits Thread
- Distill.pub
- Lilian Weng's blog

---

## 📊 Statistics

**Document Stats:**
- **Papers referenced:** 50+
- **Foundation models covered:** 8+
- **Interpretability methods:** 6+
- **Code repositories:** 10+
- **Datasets mentioned:** 15+

**Literature Review:**
- **Sections:** 10 major sections
- **Appendices:** 5 appendices
- **Tables:** 20+ comparison tables
- **Word count:** ~25,000 words

**Coverage:**
- **Years:** 2022-2026
- **Venues:** NeurIPS, ICML, ICLR, KDD
- **Domains:** Forecasting, classification, anomaly detection

---

## 🚀 Quick Links

### Essential Papers
- [TimeSAE (ICLR 2026 Workshop)](https://openreview.net/pdf?id=Ojd6YjHpyE)
- [Foundation Models Survey (KDD 2024)](https://arxiv.org/abs/2403.14735)
- [TimesFM (ICML 2024)](https://arxiv.org/abs/2310.10688)
- [Chronos (2024)](https://arxiv.org/abs/2403.07815)

### Code Repositories
- [TimesFM](https://github.com/google-research/timesfm)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Time-LLM](https://github.com/KimMeen/Time-LLM)
- [UniTS](https://github.com/mims-harvard/UniTS)

### Resources
- [FM4TS Tutorial](https://sites.google.com/view/fm4ts/home)
- [Google Research Blog](https://research.google/blog/)
- [Anthropic Interpretability](https://transformer-circuits.pub/)

---

## 📝 Changelog

### Version 1.0 (March 11, 2026)
- Initial release
- Complete literature review
- BibTeX references
- Quick start guide
- Research roadmap
- README documentation

### Planned Updates
- [ ] Add more interpretability methods
- [ ] Include more foundation models
- [ ] Add code examples with outputs
- [ ] Create video tutorials
- [ ] Add community contributions
- [ ] Update with new papers (quarterly)

---

## 💬 Final Notes

**This is a living document.** The field of mechanistic interpretability for time series foundation models is rapidly evolving. What's state-of-the-art today may be obsolete in months.

**Your contributions matter.** Whether you're adding a new paper, fixing a typo, or sharing an implementation, you're helping advance the field.

**Stay curious.** The best research comes from asking "why?" and "what if?" Keep questioning, keep exploring, keep learning.

**Build in public.** Share your findings, even if preliminary. Open science accelerates progress for everyone.

---

**Happy researching! 🚀**

*For questions, contributions, or collaborations, please open an issue or submit a pull request.*

---

*Last updated: March 11, 2026*
*Version: 1.0*
*Maintained by: [Your Name/Organization]*
