<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0
List of modified principles:
  - Changed: "Accuracy and Technical Correctness" → "Spec-driven Creation"
  - Changed: "Clarity for Learners" → "AI-assisted Precision"
  - Changed: "Modularity" → "Open-source Transparency"
  - Changed: "Practical Usefulness" → "Consistency & Maintainability"
  - Added: "Spec-driven Creation" principle
  - Added: "AI-assisted Precision" principle
  - Added: "Open-source Transparency" principle
  - Added: "Consistency & Maintainability" principle
  - Removed: "Accuracy and technical correctness", "Clarity for learners", "Modularity", "Practical usefulness", "Consistency", "Educational focus"
Added sections: Writing Style, Content Format, Technical Accuracy, Version Control Discipline, Book Structure, Deployment Requirements, Content Generation Workflow
Removed sections: Content Requirements, Output Behavior for Claude
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
  - .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs: None
-->
# AI/Spec-Driven Book Creation Constitution

## Core Principles

### I. Spec-driven Creation
All book structure, chapters, workflows, and content MUST follow Spec-Kit Plus specifications. All features and content MUST be planned, specified, and implemented using the spec → plan → analyze → implement workflow.

### II. AI-assisted Precision
Claude Code MUST be used for drafting, refactoring, and generating content while ensuring human oversight for correctness and quality. All AI-generated content MUST be reviewed and validated by human experts before acceptance.

### III. Open-source Transparency
The book MUST be fully reproducible, deployable, and maintainable via Docusaurus and GitHub Pages. All code, content, and processes MUST be open-source and transparent for community contribution and validation.

### IV. Consistency & Maintainability
All sections, components, and docs MUST follow a unified style, tone, and technical standard. The book structure MUST be maintainable and consistent across all chapters and sections.

## Key Standards

- Writing style: Clear, structured, and beginner-friendly, suitable for users learning AI/book automation workflows.
- Content format: Markdown-based Docusaurus documentation with clean folder hierarchy following Spec-Kit Plus conventions.
- Technical accuracy: All instructions for Docusaurus setup, GitHub deployment, automation, and AI workflows MUST be validated through live testing.
- Version control discipline: All generated files, specs, and updates MUST follow Git-based workflow (commit → review → push).

## Book Structure

- Book structure: Minimum 8–10 sections (Introduction, Setup, Spec Process, AI Tools, Implementation, Deployment, Case Studies, Best Practices, etc.).
- Each section MUST contain relevant subsections with clear hierarchy.
- Content MUST be organized in a logical progression suitable for learning.
- All sections MUST be cross-referenced appropriately for easy navigation.

## Technical Constraints

- Format: Markdown compatible with Docusaurus.
- MUST follow Docusaurus folder structure: `docs/` for all book content, `sidebar.js` for navigation.
- Content MUST be exportable to GitHub Pages build.
- Compatible with Claude Code workflows and Spec-Kit Plus commands.
- No proprietary content or copyrighted text without permission.

## Success Criteria

- A complete, structured Docusaurus book generated via Spec-Kit Plus workflows.
- Live deployed site on GitHub Pages without build errors.
- All specs, plans, chapters, and checklists produced using `/sp.*` commands.
- Writing is clear, accurate, and consistent across all chapters.
- Repository structure follows Spec-Kit Plus conventions.

## Content Generation Workflow

- Content generation MUST follow the spec → plan → analyze → implement workflow.
- All content MUST be created using `/sp.*` commands where applicable.
- Specifications MUST be created before implementation.
- All changes MUST be tracked through version control.

## Deployment Requirements

- Deployment: Must compile successfully on Docusaurus build and auto-deploy on GitHub Pages.
- Build process: All content MUST pass Docusaurus build without errors.
- Site performance: Pages MUST load efficiently and be mobile-responsive.
- GitHub integration: Auto-deployment workflows MUST be configured and functional.

## Governance

This constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan. All PRs/reviews MUST verify compliance. Complexity MUST be justified. All content MUST adhere to the defined principles and standards.

**Version**: 1.1.0 | **Ratified**: 2025-12-08 | **Last Amended**: 2025-12-08
