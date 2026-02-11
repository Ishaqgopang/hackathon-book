<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Modified principles: [PRINCIPLE_1_NAME] → Specification-first development, [PRINCIPLE_2_NAME] → Accuracy via official documentation, [PRINCIPLE_3_NAME] → Clarity for technical readers, [PRINCIPLE_4_NAME] → Reproducibility and deployability, [PRINCIPLE_5_NAME] → Agentic, modular design
- Added sections: Development Workflow section
- Removed sections: None
- Templates requiring updates: ✅ updated all templates
- Follow-up TODOs: None
-->
# AI/Spec-Driven Book with Embedded RAG Chatbot Constitution

## Core Principles

### Specification-first development
All features and functionality begin with detailed specifications that guide implementation. Code is written to match specifications, not the reverse. Specifications must include acceptance criteria, edge cases, and validation requirements before any implementation begins.

### Accuracy via official documentation
All technical claims, API references, and code examples must be verified against official documentation from authoritative sources. Primary sources are preferred over secondary interpretations. All third-party dependencies must be properly cited with version-specific references.

### Clarity for technical readers
Documentation and code must prioritize clarity and accessibility for technical audiences. Complex concepts should be broken down into digestible sections with practical examples. Explanations must be precise, with sufficient detail for readers to reproduce implementations.

### Reproducibility and deployability
All code examples, configurations, and deployment procedures must be tested and verified to work as documented. Development environments must be reproducible across different platforms. Deployment processes must be automated and idempotent.

### Agentic, modular design
System architecture follows modular design principles with well-defined interfaces between components. Each module should be independently testable and replaceable. Agent-based systems should maintain clear separation of concerns and predictable interaction patterns.

### Production-ready, minimal code
All implementations must be production-quality from the outset. Code should follow the principle of "minimal viable implementation" - sufficient functionality to meet requirements without unnecessary complexity. Performance and security considerations are addressed during initial development.

## Technical Standards
Technology stack requirements include Docusaurus for documentation, FastAPI for backend services, Qdrant Cloud for vector storage, and Neon Serverless Postgres for metadata. All infrastructure must support environment-variable-based secrets and be deployable to GitHub Pages with supporting cloud services.

## Development Workflow
All code changes must be accompanied by updated specifications. Pull requests require specification compliance verification. All claims in documentation must be verifiable against official sources. Code reviews check for adherence to architectural principles and proper modular design.

## Governance
This constitution supersedes all other development practices. All project activities must comply with these principles. Amendments require formal documentation and approval process. All PRs and reviews must verify constitutional compliance.

**Version**: 1.1.0 | **Ratified**: 2026-02-07 | **Last Amended**: 2026-02-07
