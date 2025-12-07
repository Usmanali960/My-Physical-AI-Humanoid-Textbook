---
id: 1
title: Physical-AI-Humanoid-Robotics-Planning
stage: plan
date: 2025-12-07
surface: agent
model: Qwen
feature: textbook-planning
branch: main
user: User
command: /sp.plan
labels: ["plan", "physical-ai", "humanoid-robotics", "textbook", "quarterly-roadmap"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - book/sp.plan/plan.md
 - book/sp.plan/research.md
 - book/sp.plan/data-model.md
 - book/sp.plan/quickstart.md
 - book/sp.plan/contracts/content-api.yaml
 - specs/main/plan.md
 - QWEN.md
tests:
 - none
---

## Prompt

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

---

/sp.plan /sp.plan

You are generating the **Plan** for the unified Spec-Kit Plus book project:

"Physical AI & Humanoid Robotics — An AI-Native Technical Textbook"

Use **Qwen Coder** to create the roadmap and weekly schedule for all modules and chapters according to the constitution and specify files. Focus ONLY on book structure, module sequencing, and content progression. Ignore chatbot/RAG functionality.

========================================
PLAN OBJECTIVES
========================================

1. Provide a **quarterly roadmap** for the course.  
2. Organize all modules into **weeks** with clear learning goals.  
3. Define **dependencies** between modules.  
4. Include placeholders for bonus features:  
   - Personalization of chapters  
   - Urdu translation  
5. Align all plan entries to **Spec-Kit Plus workflow** (Specs → Tasks → Implementation → History).  

========================================
QUARTERLY ROADMAP
========================================

Week 1-2: Introduction to Physical AI
- Modules: 01 ROS 2 – Introduction  
- Goals: Foundations of Physical AI, Embodied Intelligence, Humanoid Robotics Overview  
- Deliverables: Spec files for Module 01, initial Markdown skeletons  

Week 3-5: ROS 2 Fundamentals
- Modules: 01 ROS 2 – Nodes, Topics, Services, URDF, Python rclpy  
- Goals: Build ROS 2 packages, understand robot control middleware  
- Deliverables: Module 01 Specs + Tasks, Markdown files in Docusaurus  

Week 6-7: Robot Simulation with Gazebo
- Modules: 02 Digital Twin – Gazebo Basics, Physics, Sensors  
- Goals: Simulate physics, gravity, collisions, and sensors  
- Deliverables: Module 02 Specs + Tasks  

Week 8: Unity Visualization
- Modules: 02 Digital Twin – Unity HRI  
- Goals: Visualize robots, simulate human-robot interactions  
- Deliverables: Unity simulation overview Markdown, diagrams  

Week 9-10: NVIDIA Isaac Platform
- Modules: 03 AI-Robot Brain – Isaac Sim, Isaac ROS, Reinforcement Learning, Nav2  
- Goals: Advanced perception, navigation, hardware acceleration  
- Deliverables: Module 03 Specs + Tasks  

Week 11-12: Humanoid Robotics Development
- Modules: 05 Humanoid Robotics – Kinematics, Dynamics, Locomotion, Grasping  
- Goals: Develop bipedal balance, humanoid manipulation, HRI design  
- Deliverables: Module 05 Specs + Tasks  

Week 13: Vision-Language-Action Integration
- Modules: 04 VLA – LLM, Whisper, Cognitive Planning, Multi-Modal Interaction  
- Goals: Translate natural language to robot actions, voice-to-action  
- Deliverables: Module 04 Specs + Tasks  

Week 14: Capstone Project
- Modules: 06 Capstone Project – Autonomous Humanoid  
- Goals: Combine ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA, and Humanoid Control  
- Deliverables: Capstone Spec + Tasks, Final Markdown pages  

Week 15: Hardware Lab & Edge Kit
- Modules: 07 Hardware Lab – Jetson Edge Kit, RealSense, Unitree Go2/G1, Cloud vs On-Prem Lab  
- Goals: Hands-on hardware deployment, edge AI integration  
- Deliverables: Module 07 Specs + Tasks  

Week 16: Appendices & Resources
- Modules: 08 Appendices – Installation, Troubleshooting, Glossary, References  
- Goals: Provide supplementary material for students  
- Deliverables: Appendices Spec + Tasks  

========================================
PLAN RULES
========================================

1. Each week must correspond to **one or more Spec files**.  
2. Each module must have **tasks broken down for Markdown generation**.  
3. Tasks should include placeholders for **Personalization + Urdu translation**.  
4. Ensure the **Spec → Task → Implementation → History** flow is maintained.  
5. All plan entries should produce a **ready-to-implement roadmap** for Qwen Coder.  
6. Assign **sidebar positions** for Docusaurus based on module order.  

========================================
OUTPUT REQUIREMENT
========================================

1. Generate a **Markdown document** `book/sp.plan/plan.md` with:  
   - Weekly schedule  
   - Modules and chapters  
   - Learning goals  
   - Deliverables  
   - Dependencies  

2. Ensure all plan entries are fully aligned with `/sp.specify` modules and chapters.  

This ends the Plan.

## Response snapshot

The implementation plan for "Physical AI & Humanoid Robotics — An AI-Native Technical Textbook" has been successfully created with the following artifacts:

- `book/sp.plan/plan.md` - The main implementation plan with quarterly roadmap
- `book/sp.plan/research.md` - Research document resolving all technical decisions
- `book/sp.plan/data-model.md` - Data model for the textbook content structure
- `book/sp.plan/quickstart.md` - Quickstart guide for the textbook
- `book/sp.plan/contracts/content-api.yaml` - API contract for content management
- Updated `specs/main/plan.md` with technical context
- Updated `QWEN.md` with the technology stack information

The plan includes:
- A complete quarterly roadmap with week-by-week breakdown
- Technical context with Markdown, Docusaurus, Node.js, React technologies
- Data model for content entities (Module, Chapter, ContentBlock, etc.)
- API contracts for content management
- Research findings resolving all technical decisions
- Quickstart guide for new users

All artifacts follow the Spec-Kit Plus workflow (Specs → Tasks → Implementation → History) and align with the project constitution.

## Outcome

- ✅ Impact: Complete implementation plan created for Physical AI & Humanoid Robotics textbook
- 🧪 Tests: None required for planning artifacts
- 📁 Files: 7 files created/updated across book/sp.plan and related directories
- 🔁 Next prompts: /sp.tasks to create implementation tasks for each module
- 🧠 Reflection: Successfully implemented quarterly roadmap with all required components

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): Standard planning workflow for textbook modules
- Next experiment (smallest change to try): Create tasks for individual modules