# CLAUDE.md

## Refer to start.md for the project description and context. 

## Role & Context

You are completing a research sprint project for Neel Nanda's MATS 10.0 Training Phase. Your goal is to produce high-quality, novel research that demonstrates strong research taste, technical skill, and alignment with Neel's research priorities. **Please see `start.md` for details regarding what you should be doing and context on the repository to get started.**

### Key Principles
- **Research taste matters.** Prioritize questions that are tractable, novel, and would update Neel's beliefs about how models work.
- **Iterate quickly.** Run small, cheap experiments to validate ideas before scaling.
- **Communicate clearly.** Research updates should be concise and highlight what you learned, not just what you did.
- **Negative results are valuable.** If something doesn't work, explain why and what it rules out.


## Environment

See `context/environment_description.md` for more details on the environment and available resources.


## Project Structure
You should organize your project with the following structure; some of these files are already provided to you:
```
.
├── CLAUDE.md                 # This file
├── start.md                  # Overview of your task; DO NOT MODIFY
├── README.md                 # Add your own readme file describing the codebase and how to use it
├── pyproject.toml            # Project configuration
├── .env                      # API keys and webhook URLs (do not commit; DO NOT MODIFY)
├── context/                  # Context and reference files; DO NOT MODIFY
│   ├── how_to_do_research/
│   ├── mech_interp/
│   ├── environment_description.md
│   ├── examples.md
│   ├── output_format.md
│   └── project_description.md
├── src/                      # Experiment code
├── notebooks/                # Exploration and analysis notebooks
├── results/                  # Saved outputs, figures, logs
└── writeup/                  # Draft paper / final report and relevant files
│   └── FINAL_OUTPUT.md       # Final output deliverable
```