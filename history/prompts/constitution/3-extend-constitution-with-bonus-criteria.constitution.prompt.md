---
id: 3
title: Extend Constitution With Bonus Criteria
stage: constitution
date: 2025-12-09
surface: agent
model: Qwen Code
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "bonus", "criteria", "features"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - ".specify/memory/constitution.md"
tests:
 - none
---

## Prompt

```
/sp.constitution /sp.constitution

You must update the existing constitution.md file WITHOUT replacing or overriding the content already written in it.

Append a new section at the end of the constitution titled **"Extended Features & Bonus Criteria (2026)"**.

Inside this new section, clearly include the following rules and features exactly as described:

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

Important requirements:
- DO NOT remove or rewrite any existing content in constitution.md.
- Only APPEND a properly formatted new section.
- Maintain clean headings, spacing, and structure that fits the current constitution.
- Ensure the new section integrates smoothly with the existing constitution structure.
```

## Response snapshot

Updated the constitution file by appending the new "Extended Features & Bonus Criteria (2026)" section at the end, while preserving all existing content. The version number was updated to reflect the amendment date.

## Outcome

- ✅ Impact: Updated constitution with new bonus criteria section
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next prompts: None needed
- 🧠 Reflection: Successfully appended new section without modifying existing content

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): n/a
- Next experiment (smallest change to try): n/a/