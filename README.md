# Domain-MOE 
> ”Domain-adaptive routing in Mixture of Expert models improves robustness across domains in earth observation”

Basically: one expert model per domain (country). Gating network learns to map an unlabeled (no domain label) input to its respective expert model. We explicity train the gating network to do so in a supervised manner using a combined loss. The key here is that the experts will share some weights and therefore representations learned from one country (i.e., more resources/data) can be beneficial for another country model (less resources/data).

## Models to run 
We will start experimentation with the skin cancer dataset from 221 since it has well defined domains and is a simple task. 
- Supervised baseline 
- Supervised baseline per-group (individually train a model per-group) 
- Standard MOE approach
- Our proposed MOE approach with tailored domain routing 


## Papers
- Multi-Source Domain Adaptation with Mixture of Experts
  - Basically do something really similar 
- Towards Understanding the Mixture-of-Experts
  Layer in Deep Learning
- SPARSE MIXTURE-OF-EXPERTS ARE DOMAIN GENERALIZABLE LEARNERS
- Meta-DMoE: Adapting to Domain Shift by
  Meta-Distillation from Mixture-of-Experts


## Modeling 


| Method   | Overall Accuracy | Skin Type's {1, 2, 3} Accuracy | Skin Type's {4, 5, 6} Accuracy | Logs |
|---------|---|---|---| --- |
| Supervised baseline | 0% | 0% | 0% | n/a |
