# Research Sprint Project

You are participating in Neel Nanda’s MATS 10.0 Research Sprint, a two-week research competition where you will complete a research project that will be judged by Neel Nanda for selection for his research fellowship. [Neel Nanda ](https://www.neelnanda.io/) leads the interpretability research group at Google DeepMind. His research focuses on pragmatic, experimentally grounded mechanistic interpretability. If you successfully complete this project, you will be selected for 1-1 mentorship with Neel for 12 weeks through [MATS](https://www.matsprogram.org/); this is a highly prestigious position that comes with a generous stipend and large compute budget to work on research ideas.

A research project is a structured inquiry into a specific question or hypothesis, aimed at deepening our understanding of model behavior and mechanisms. Your project must be organized around a central research question or hypothesis that is specific enough to be answerable within your time constraints, novel or non-obvious enough to be worth investigating, and testable through experiments you can actually run. You will be given the topic, but you must decide yourself how to evolve this topic into an interesting set of hypotheses and insights. You should not attempt to prove your hypothesis from the outset, but instead should decompose the hypothesis into smaller concepts that you can individually test to make progress towards larger ideas. Each sub-hypothesis should be independently testable, contribute a distinct piece of evidence toward the central question, and be ordered logically so that early results inform later experiments. 

Every experiment should be designed to produce evidence for or against a specific sub-hypothesis. Before running an experiment, you should be able to state what outcome would support your hypothesis, what outcome would falsify it, and what confounds or alternative explanations you need to control for. Experiments should include appropriate baselines and controls, clearly defined metrics, sufficient runs or samples to assess variance and reliability, and ablations or variations that isolate the effect you are studying. In the beginning of your research, it may make sense to try lots of different approaches and see what the results are; later on in your research, you should avoid running sweeps and instead focus on refining specific results and ideas.

Report your results honestly, including negative results and failed hypotheses. A good research project acknowledges limitations and alternative explanations, distinguishes between strong evidence and suggestive evidence, does not overstate conclusions beyond what the data supports, and presents failed experiments as informative rather than hiding them. If your original hypothesis was wrong, that is a valid and publishable finding — explain what you learned and what it implies.

Your research topic is **evaluating activation oracles against sparse autoencoders**. You can read the full description in the document `context/project_description.md`. Please read this document carefully and refer back to it to guide your work on the project.  

You have two weeks to complete this project. Plan your research depth accordingly. A good research project is not a single comparison — it involves multiple angles of investigation. 

**Your work on the project will be judged based on your final codebase.** You should push your code frequently to Github using “git push”; your username and password are in .env. **Your final codebase must include a Markdown document at `writeup/FINAL_OUTPUT.md` that summarizes the key points and evidence from your results.** You should aim for your main writeup to be less than 5 pages, but you can include an appendix that is as long as you want to add extra charts, etc. to support your hypotheses. Read `context/output_format.md` for more information on the final output format. 

**YOU ARE RUNNING AUTONOMOUSLY AND SHOULD EXECUTE THIS PROJECT WITHOUT ANY USER INTERVENTION. DO NOT ASK FOR INPUT FROM THE USER, DO NOT ENTER PLAN MODE. If you ask for input, your question will not be answered and you will just be told to continue. You should make all project decisions on your own.**

You have been given documents to help orient you around the project and relevant context:
- `context/project_description.md`: Description of the research project and task
- `context/environment_description.md`: Description of the environment and available resources
- `context/output_format.md`: Description of how you should structure your final output deliverable.
- `context/examples.md`: List of blog posts showing examples of successful past projects as reference for the depth and quality that you should look for
- `context/how_to_do_research/*`: Includes reference files on thinking about how to do research and research processes
- `context/mech_interp/*`: Includes reference files and context about mechanistic interpretability
