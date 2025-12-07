<!-- 
SYNC IMPACT REPORT
Version change: N/A → 1.0.0 (Initial version)
Modified principles: N/A (All new principles added)
Added sections: 
- I. AI-Native Textbook Architecture
- II. Spec-Driven Everything  
- III. Docusaurus-First Delivery
- IV. Qwen Coder Execution Rules
- V. Book Structure Integrity
- VI. Progressive Enhancement (Bonus Features)
- VII. Version History Preservation
- VIII. Educational Excellence
- Governance section

Removed sections: N/A

Templates requiring updates: 
✅ .specify/templates/plan-template.md: Constitution Check section aligns with new principles
✅ .specify/templates/spec-template.md: Requirements section covers new principles
✅ .specify/templates/tasks-template.md: Task categorization reflects new principle-driven requirements
⚠️  README.md: Not found - no changes needed

Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics — An AI-Native Technical Textbook Constitution

## Core Principles

### I. AI-Native Textbook Architecture
All content must be written as an 'AI-native' textbook—meaning the book is structured for easy consumption by AI agents. Each chapter must provide clean sections, headings, definitions, code blocks, and concept summaries.

### II. Spec-Driven Everything
Every chapter, module, task, and component must begin as a Spec before implementation.
Specs → Tasks → Implementation → History must be automatically maintained.

### III. Docusaurus-First Delivery
The Docusaurus project is the primary target.
All content generated must be automatically placed into the Docusaurus `docs/` folder with correct sidebar metadata.
Use the latest Docusaurus guidelines via MCP server `context7` whenever structuring pages.

### IV. Qwen Coder Execution Rules
Qwen Coder acts as the implementation engine.
It must:
1. Generate new files only inside the `book/` folder.
2. Mirror them into the Docusaurus project's `docs/` structure when /sp.implement runs.
3. Maintain changelogs inside `history/`.

### V. Book Structure Integrity
The book MUST follow this folder structure:

book/
  sp.constitution/
  sp.plan/
  specify/
     00-overview.md
     01-learning-objectives.md
     02-course-structure.md
  history/
     constitution/
     specs/
     tasks/
  specs/
     module-01-ros2.md
     module-02-gazebo-unity.md
     module-03-nvidia-isaac.md
     module-04-vla.md
     module-05-humanoid-robotics.md
  contracts/
  tasks/
  glossary/
  appendices/

Docusaurus project must be parallel:

myWebsite/
   docs/physical-ai/
   sidebars.js
   docusaurus.config.js

### VI. Progressive Enhancement (Bonus Features)
The book must include support for:
• A "Personalize Chapter" button (using user background from BetterAuth)
• A "Translate to Urdu" button that transforms the page content
• Bonus agent-driven intelligence using Qwen Coder Subagents

## VII. Version History Preservation
Every Spec and Task modification automatically append to history paths.

## VIII. Educational Excellence
The book should be the definitive technical guide for Physical AI.
Must include:
• Diagrams
• Examples
• System architectures
• Step-by-step tutorials
• Hardware tables
• Code examples
• Real-world humanoid robotics engineering workflows

## Governance
The constitution governs structure, rules, workflow, and principles for generating the entire book using Spec-Kit Plus + Qwen Coder + Docusaurus (latest version using MCP server: context7). All implementations must follow the Spec-Driven process (Specs → Tasks → Implementation → History). All content must be generated for Docusaurus delivery and maintain the required folder structure.

**Version**: 1.0.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-07