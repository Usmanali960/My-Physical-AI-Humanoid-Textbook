---
id: constitution
title: Constitution - Physical AI & Humanoid Robotics
sidebar_label: Constitution
---

# Physical AI & Humanoid Robotics — An AI-Native Technical Textbook Constitution

## Core Principles

### I. AI-Native Textbook Architecture
All content must be written as an "AI-native" textbook—meaning the book is structured for easy consumption by AI agents. Each chapter must provide clean sections, headings, definitions, code blocks, and concept summaries.

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

**Version**: 1.0.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-09

## Extended Features & Bonus Criteria (2026)

3. Participants will receive points out of 100 for the base functionality already defined in the constitution. This includes all core requirements previously listed.

4. Participants can earn up to 50 extra bonus points by creating and using **reusable intelligence** via:
   - Claude Code Subagents
   - Agent Skills
   These must be integrated inside the online book project.

5. Participants can receive up to 50 extra bonus points if they implement **Signup and Signin** using **Better Auth (https://www.better-auth.com/)**.
   At signup, the system must ask questions about the user's software and hardware background.
   This data must later be used to personalize content inside the book.

6. Participants can receive up to 50 extra bonus points if logged-in users can **personalize the content of each chapter** by pressing a button at the start of the chapter.

7. Participants can receive up to 50 extra bonus points if logged-in users can **translate any chapter content into Urdu** by pressing a button at the start of the chapter.
