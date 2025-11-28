REFERENCES: 
- [Cognition -- Don't Build Multi-agents](https://cognition.ai/blog/dont-build-multi-agents) (aka context + decision stuff)
- [Meta Agents Research Environment](https://github.com/facebookresearch/meta-agents-research-environments.git)
    Citation:
```
@misc{andrews2025arescalingagentenvironments,
    title={ARE: Scaling Up Agent Environments and Evaluations},
    author={Pierre Andrews and Amine Benhalloum and Gerard Moreno-Torres Bertran and Matteo Bettini and Amar Budhiraja and Ricardo Silveira Cabral and Virginie Do and Romain Froger and Emilien Garreau and Jean-Baptiste Gaya and Hugo Laurençon and Maxime Lecanu and Kunal Malkan and Dheeraj Mekala and Pierre Ménard and Grégoire Mialon and Ulyana Piterbarg and Mikhail Plekhanov and Mathieu Rita and Andrey Rusakov and Thomas Scialom and Vladislav Vorotilov and Mengjue Wang and Ian Yu},
    year={2025},
    eprint={2509.17158},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2509.17158},
}   
```

JACK -- LIT REVIEW:
- [Google: CoA](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)
    - Very raw, more focussed on long-context than sharing information; traditional text-based information sharing
    - Orchestrator ("manager") and worker agents to parse chunks
    - Independence vs. shared knowledge: it's helpful for agents to be able to explore independently, but they need to be able to forward/share info
** - [SagaLLM](https://dl.acm.org/doi/pdf/10.14778/3750601.3750611)
    - will be useful re: what to include in the metadata for each of the episodes
- [CA-MAS survey](https://arxiv.org/pdf/2402.01968)

---

# 28 Nov
NEXT:
- [ ] finish up evals framework + scale up evals (1-2h_)
- [ ] implement base multi-agent system (1h)
- [ ] add Basis contextualizer/ID system + fine-tune (3h)
- [ ] run evals (literally 1 command)
- [ ] based on evals and stats, generate plots (idk)
- [ ] writeup