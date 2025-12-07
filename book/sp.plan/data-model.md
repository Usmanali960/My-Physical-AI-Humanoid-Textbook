# Data Model: Physical AI & Humanoid Robotics — An AI-Native Technical Textbook

**Created**: 2025-12-07
**Feature**: Physical AI & Humanoid Robotics Textbook
**Plan**: book/sp.plan/plan.md

## Overview

This document defines the data model for the Physical AI & Humanoid Robotics textbook project. It represents the content structure as entities with attributes and relationships, following the requirements from the original specification.

## Content Entities

### Module
**Description**: The primary organizational unit of the textbook
**Attributes**:
- moduleId: String (e.g., "module-01-ros2")
- title: String (e.g., "ROS 2: The Robotic Nervous System")
- description: String (brief overview of the module content)
- duration: Number (estimated weeks to complete)
- learningObjectives: Array[String] (what students should learn)
- prerequisites: Array[String] (knowledge required before starting)
- dependencies: Array[Module] (modules that must be completed first)
- chapters: Array[Chapter] (list of chapters in the module)
- sidebarPosition: Number (position in the navigation sidebar)

**Validation rules**:
- moduleId must follow the pattern "module-[0-9]{2}-[a-z-]+"
- title must be between 5 and 100 characters
- duration must be between 1 and 4 weeks

### Chapter
**Description**: A subunit within a module
**Attributes**:
- chapterId: String (e.g., "module-01-chapter-01")
- title: String (e.g., "Introduction to ROS 2")
- description: String (brief overview of the chapter content)
- estimatedTime: String (e.g., "2 hours", "3 lectures")
- content: ContentBlock[] (structured content of the chapter)
- parentModule: Module (reference to the parent module)
- position: Number (position within the parent module)

**Validation rules**:
- chapterId must follow the pattern "module-[0-9]{2}-chapter-[0-9]{2}"
- title must be between 5 and 100 characters
- estimatedTime must follow the format "[number] [unit]" where unit is hours/lectures/days

### ContentBlock
**Description**: A unit of content within a chapter
**Attributes**:
- blockId: String (unique identifier for the block)
- type: String (e.g., "text", "code", "diagram", "exercise", "example")
- content: String (the actual content)
- headingLevel: Number (for text blocks, represents H1-H6)
- language: String (for code blocks, represents the programming language)
- title: String (optional title for the block)

**Validation rules**:
- type must be one of the predefined values
- headingLevel must be between 1 and 6 for text blocks
- language must be a valid programming language for code blocks

### LearningObjective
**Description**: A specific learning outcome
**Attributes**:
- objectiveId: String (unique identifier)
- description: String (what the student should be able to do)
- difficultyLevel: String ("beginner", "intermediate", "advanced")
- assessmentCriteria: String (how the objective will be measured)

**Validation rules**:
- difficultyLevel must be one of the predefined values
- description must be between 10 and 200 characters

### Resource
**Description**: Supporting material for modules/chapters
**Attributes**:
- resourceId: String (unique identifier)
- title: String (name of the resource)
- type: String ("video", "tutorial", "code", "paper", "tool", "diagram")
- url: String (location of the resource)
- description: String (what the resource provides)
- parentEntity: Module|Chapter (what content the resource supports)

**Validation rules**:
- type must be one of the predefined values
- url must be a valid URL or file path

### Assessment
**Description**: Evaluation mechanism for content
**Attributes**:
- assessmentId: String (unique identifier)
- title: String (name of the assessment)
- type: String ("quiz", "exercise", "project", "simulation")
- difficultyLevel: String ("beginner", "intermediate", "advanced")
- questions: Array[Question] (list of questions in the assessment)
- parentEntity: Module|Chapter (what content is being assessed)

**Validation rules**:
- type must be one of the predefined values
- difficultyLevel must be one of the predefined values

### Question
**Description**: Individual question in an assessment
**Attributes**:
- questionId: String (unique identifier)
- text: String (the question text)
- type: String ("multiple-choice", "short-answer", "coding", "essay")
- options: Array[String] (for multiple-choice questions)
- correctAnswer: String|Array (the correct answer)
- explanation: String (why the answer is correct)

**Validation rules**:
- type must be one of the predefined values
- options must be non-empty for multiple-choice questions

## Entity Relationships

### Module ↔ Chapter
- One module contains many chapters (1 to many)
- Each chapter belongs to exactly one module
- When a module is removed, its chapters are also removed

### Module ↔ LearningObjective
- One module has many learning objectives (1 to many)
- Each learning objective belongs to exactly one module

### Chapter ↔ ContentBlock
- One chapter contains many content blocks (1 to many)
- Each content block belongs to exactly one chapter

### Module ↔ Resource
- One module has many resources (1 to many)
- Each resource belongs to exactly one module (or chapter)

### Module ↔ Assessment
- One module has many assessments (1 to many)
- Each assessment belongs to exactly one module (or chapter)

### Assessment ↔ Question
- One assessment contains many questions (1 to many)
- Each question belongs to exactly one assessment

## State Transitions

### Content Development States
- `draft`: Content has been planned but not yet created
- `in-progress`: Content is being actively created
- `review`: Content has been created and is awaiting review
- `approved`: Content has passed review and is ready for publication
- `published`: Content is live in the textbook
- `deprecated`: Content is outdated and should be updated

**Valid transitions**:
- `draft` → `in-progress`
- `in-progress` → `review`
- `review` → `in-progress` (if changes requested)
- `review` → `approved`
- `approved` → `review` (for further changes)
- `approved` → `published`
- Any state → `deprecated`

## Indexes for Performance

### Module
- moduleId (unique)
- sidebarPosition
- title

### Chapter
- chapterId (unique)
- parentModule
- position

### ContentBlock
- blockId (unique)
- parentChapter
- type

## Constraints

1. Each Module must have at least one Chapter
2. Each Chapter must have at least one ContentBlock
3. Module duration cannot exceed 4 weeks
4. ContentBlock content must be between 10 and 10,000 characters
5. Each Assessment must have at least one Question
6. Chapter position within a module must be unique
7. Module sidebar positions must be consecutive and start from 1

## Sample Data

```
Module {
  moduleId: "module-01-ros2",
  title: "ROS 2: The Robotic Nervous System",
  description: "Introduction to ROS 2 fundamentals and architecture",
  duration: 3,
  learningObjectives: [
    "Understand ROS 2 architecture and core concepts",
    "Implement basic publisher/subscriber patterns",
    "Use rclpy for Python ROS 2 development"
  ],
  prerequisites: [
    "Basic Python programming",
    "Linux command line familiarity"
  ],
  dependencies: [],
  chapters: [
    "module-01-chapter-01",
    "module-01-chapter-02", 
    "module-01-chapter-03",
    "module-01-chapter-04"
  ],
  sidebarPosition: 1
}

Chapter {
  chapterId: "module-01-chapter-01",
  title: "Introduction to ROS 2",
  description: "Understanding the fundamentals of the Robot Operating System",
  estimatedTime: "2 hours",
  content: [
    "content-block-001",
    "content-block-002",
    "content-block-003"
  ],
  parentModule: "module-01-ros2",
  position: 1
}
```

## API Endpoints (for potential future development)

These represent potential endpoints for managing the textbook content:

### GET /api/modules
- Retrieve all modules with basic information
- Query parameters: `status` (filter by development state), `limit`, `offset`

### GET /api/modules/{moduleId}
- Retrieve detailed information about a specific module
- Includes all chapters and learning objectives

### GET /api/chapters/{chapterId}
- Retrieve detailed content of a specific chapter
- Includes all content blocks

### POST /api/modules
- Create a new module
- Requires authentication and authorization

### PUT /api/modules/{moduleId}
- Update an existing module
- Requires proper permissions

### DELETE /api/modules/{moduleId}
- Mark a module as deprecated
- Requires proper permissions