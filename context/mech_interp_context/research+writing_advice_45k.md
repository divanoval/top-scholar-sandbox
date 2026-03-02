# [How I Think About My Research Process: Explore, Understand, Distill](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand)

by [Neel Nanda](https://www.alignmentforum.org/users/neel-nanda-1?from=post_header)26th Apr 2025*This is the first post in a sequence about how I think about and break down my research process. Post 2 is coming soon.*

*Thanks to Oli Clive-Griffin, Paul Bogdan, Shivam Raval and especially to Jemima Jones for help and feedback, and to my co-author Gemini 2.5 Pro - putting 200K tokens of past blog posts and a long voice memo in the context window is OP.*

## Introduction

Research, especially in a young and rapidly evolving field like mechanistic interpretability (mech interp), can often feel messy, confusing, and intimidating. Where do you even start? How do you know if you're making progress? When do you double down, and when do you pivot?

These are far from settled questions, but I’ve supervised 20+ papers by now, and have developed my own mental model of the research process that I find helpful. This isn't *the* definitive way to do research (and I’d love to hear other people’s perspectives!) but it's a way that has worked for me and others.

My goal here is to demystify the process by breaking it down into stages and offering some practical advice on common pitfalls and productive mindsets for each stage. I’ve also tried to be concrete about what the various facets of ‘being a good researcher’ actually mean, like ‘research taste’ ([see post 3](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research)). I’ve written this post for a mech interp audience, but hopefully it is useful for any empirical science with short feedback loops, and possibly even beyond that.

This guide focuses more on the *strategic* (high-level direction, when to give up or pivot, etc) and *tactical* (what to do next, how to prioritise, etc) aspects of research – the "how to think about it" rather than just the "how to do it." Some of skills (coding, reading papers, understanding ML/mech interp concepts) are vital for how to do it, but not in scope here (I recommend the [ARENA curriculum](https://arena-chapter1-transformer-interp.streamlit.app/) and [my paper reading list](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite) if you need to skill up).

How to get started? Strategic and tactical thinking are hard skills, and it is rare to be any good at them when starting out at research (or ever tbh). The best way to learn them is by trying things, making predictions, seeing what you get right or wrong (i.e., getting feedback from reality), and iterating. Mentorship can substantially speed up this process by providing ["supervised data" to learn from](https://colah.github.io/notes/taste/), but either way you ultimately learn by doing.

I’ve erred towards making this post comprehensive, which may make it somewhat overwhelming. You do *not*need to try to remember everything in here! Instead think of it more as a guide for the high level things to keep in mind, and a source of advice for what to do at each stage. And, obviously, this is massively flavoured by my own subjective experience and may not generalise to you - I’d love to hear what other researchers think.

**A cautionary note:** Research is hard. Expect frustration, dead ends, and failed hypotheses. Imposter syndrome is common. Focus on the process and what you're learning. Take breaks, the total change to productive time is typically positive. Find sustainable ways to work. [Your standards are likely too high](https://www.neelnanda.io/blog/35-standards).

## The key stages

I see research as breaking down into a few stages:

1. **Ideation - Choose a problem/domain to focus on**
2. **Exploration - Gain Surface area**
	1. **North star**: Gain information
3. **Understanding - Test Hypotheses**
	1. **North star**: Convince *yourself*of a key hypothesis
4. **Distillation - Compress, Refine, Communicate**
	1. **North star**: Compress your research findings into concise, rigorous truth that you can communicate to the world

### Ideation (Stage 1): Choose a problem

- This can vary from a long, high-effort exploration across areas looking for a promising angle, to just being handed a problem by a mentor.
	- Replicating and extending an existing paper can be a good starting point, especially if you don’t have an existing mentor.
- This stage is crucial, but if you have a mentor (or other high quality source of suggestions, like someone else’s research agenda) it can be quick to just lean on them.
- It's important to understand how your work fits into the existing literature: what is already known about the problem and what remains open.
	- Where possible, for your first project or two, lean on a mentor for guidance and just read a few key papers. Building deep knowledge of a literature takes time, and is easier once you have some hands-on experience.
	- Google/OpenAI Deep Research is invaluable for literature reviews, especially in unfamiliar domains.
- Doing this well yourself and choosing a good problem often requires "**research taste**", and is the most commonly discussed aspect, but [is **just one facet of what research taste means**](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research) - research taste also covers the following:
	- Exploration: **Noticing** **when an anomaly is interesting** and should be investigated, vs boring and to be ignored
	- Understanding: **Designing great experiments** that precisely distinguish hypotheses. This often stems from having a deep enough conceptual understanding to intuit *why*a hypothesis is true
	- Distillation: Having the taste to **identify the most interesting and defensible narrative**, and what to deprioritise.
	- On a broader level, I see research taste as being about an intuitive understanding of what good research looks like, to both guide high level strategy and tactical decisions in practice, informed by a deep understanding of the domain, familiarity with what good and bad research looks like, and the high level strategic picture of which problems actually matter.

### Exploration (Stage 2): Gain surface area

- *Examples:*[*My research streams*](https://www.youtube.com/watch?v=m8tzXelUTLo&list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T)*, and*[*my Othello research process write-up*](https://www.alignmentforum.org/s/nhGNHyJHbrofpPbRG/p/TAz44Lb9n9yf52pv8)
- At the start, your understanding of the problem is often vague. Naively, it’s easy to think of research as being about testing specific hypotheses, but in practice you often start out not even knowing the right questions to ask, or the most promising directions. The exploration stage is about moving past this.
	- E.g. starting with “what changes in an LLM during chat fine-tuning?” or even “I’m sure there’s something interesting about how chat models behave, let’s mess around and find out”
- **Your north star is just to gain information** - do exploratory experiments, visualise data, follow your curiosity, prioritise moving fast.
- Junior researchers often get stuck in the early stages of a project and don’t know what to do next. In my opinion this is because **they** **think they are in the understanding stage, but are actually in the exploration stage**.
	- That is, they think they ought to have a clear goal, and hypothesis, and obvious next step, and feel bad when they don’t. But this is totally fine and normal!
	- The solution is to have a toolkit of standard ways to gain surface area, brainstorm experiments that might teach something interesting, and be comfortable exploring a bunch and hoping something interesting happens.
- Not having a clear goal/next step doesn’t mean that you don’t need to prioritise! **Prioritise for information gain**.
	- Try to do a lot of experiments (and don’t be a perfectionist about finding the ‘best’ experiments!), visualise things in many different ways, ensure you’re always learning.
	- Frequently ask yourself “**am I getting enough information per unit time**?” If you haven’t learned anything recently, shake it up.
	- Having fast feedback loops and powerful, flexible tooling is absolutely crucial here.
- Note: **In the long-term exploration should feel like play** - be fascinated by a problem, follow your curiosity, try to understand it deeply, zooming out when you get bored, etc (though it's still worth checking in on whether you're in a rabbit hole). But this isn't something you should worry about at first, as it needs well calibrated intuitions, which take time.
- Note: often most of the work in the exploration was about **discovering the right kinds of questions to be asking**, e.g. that where information was stored is an important and interesting question, crystallising that into a precise hypothesis is often easy after that.
	- This both means ‘identify the right questions to ask’, but also gain a **deeper understanding and intuition of the domain** so you can design experiments that make sense, and **build a more gears-level model** of why a certain question may or may not be true.
- A key practical tip is to **keep a highlights doc** of particularly interesting results, this makes it easier to spot connections

### Understanding (Stage 3): Test Hypotheses

- This stage begins when you understand the problem domain enough to **have some specific hypotheses that you think are interesting** - hypotheses you can write down, and have some idea of what evidence you could find to show if they’re true or false.
	- E.g. “do chat models store summarised information about the user prompt in the <end\_of\_turn> special token?”
- Your north star is to **gain evidence for and against these hypotheses**
	- Here the prioritisation is a mix of goal-directed and exploratory - you often need to briefly dip back into explore mode as you realise your hypothesis was ill-posed, your experiment didn’t make sense, you get weird and anomalous results, etc.
	- This stage is much closer to what people imagine when thinking about research.
	- Frequently ask yourself “**what am I learning and is it relevant?**”
- The mark of a good researcher is a deep commitment to **skepticism of your results**.
	- You’ll have hypotheses that are wrong, experiments that are inconclusive, beautiful methods that lose to dumb baselines, etc. This is totally fine and normal, and a part of the natural process of science, but emotionally can be pretty hard to accept.
	- This *sounds*obvious, but in practice this requires constant active effort, and if you are not actively doing this you’ll inevitably fall into traps. Always seek alternative explanations, seek and implement strong baselines, check for bugs, etc.
- A surprisingly deep and nuanced skill is **designing good experiments**. I think of this as one facet of “[research taste](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research)”
	- A great experiment elegantly, and conclusively distinguishes between several plausible hypotheses, validates non-trivial predictions made by one hypothesis, and is tractable to implement in practice.
		- This is an ideal rarely reached in practice but helpful to have in mind
	- My internal experience when generating good experiments is often that I try to simulate the world where hypothesis X is true, think through what this would mean and all the various implications of this, and notice if any can be turned into good experiments.
	- When reading papers, pay attention to the key experiments that their core claims hinge upon and ask yourself what made it important and how you might've thought of that experiment.

### Distillation (Stage 4): Compress, Refine, Communicate

- This stage begins when you have **enough evidence for*****you*****to be fairly convinced that your hypotheses are true/false**
- The north star here is to **distill your research findings**into**concise, rigorous truth**that you can**communicate to the world**
	- **Compress**your work into some concrete, well-scoped claims - something you could list in a few bullet points. Compress it as far as you can without losing the message. Readers will not take away more than a few claims.
		- How would you explain your work to a peer? How would you write a lightning talk?
	- **Refine**your evidence into a rigorous case for each key claim, enough to be persuasive to a skeptical observer
		- This is persuasive in the sense of “actually provide strong evidence”, not just writing well enough that people don’t notice flaws! This means sanity checks, statistical robustness, and strong baselines.
		- Note that this is a higher bar than convincing yourself, both since you’re aiming for a more skeptical observer and you need to make all the key evidence you’ve seen legible to an outsider.
		- You should spend a lot of time on red-teaming here - what could you be missing? What alternative hypotheses could explain your observations? What experiments could distinguish between them? Etc
	- **Communicate**these with a clear and concise write-up - make clear what your points are, what evidence you provide, and its limitations. Write to inform, not persuade - if you are clear (a high bar), and your results are interesting, people will likely appreciate your work.
		- The form of write-up doesn’t really matter - Arxiv paper, blog post, peer-reviewed paper, etc. It doesn’t need to be polished, it just needs to present the evidence clearly, and to have strong enough evidence to meaningfully inform someone’s opinion
- **People often under-rate this stage** and think doing the write-up is wasting time better spent on research, and can be left to the last minute. I think it’s actually a great use of time, at least for the first draft! I typically recommend my scholars make a start on distillation a month before conference deadlines.
	- Writing things up forces you to clarify your understanding to yourself. You also often notice holes and missing experiments. A common anecdote is that people didn’t really understand their project until they wrote it up.
	- If you don’t communicate your research well, it’s very hard to have an impact with it! (or to get recognition and career capital)
- Conversely, **people often over-rate this stage**and default to writing a paper with the main goal of getting accepted to a conference. This has obvious advantages, but can also lead to warped thinking if you’re thinking about it from the start.
	- E.g. choosing questions that look good rather than being important, or focusing on forms of evidence that reviewers will like or understand, rather than ruthlessly focusing on actually establishing what’s true.
- Sometimes you’ll discover that actually things are way messier than thought. It’s important to acknowledge this, rather than denying inconvenient truths! **Your ultimate goal is to find truth, not to produce an exciting paper**. You may need to go back to understanding or even exploration - this is totally fine and normal, and does not mean you’ve screwed anything up.

*Next up:*[*Post 2 of the sequence*](https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking)*, on key research mindsets*

# [My Research Process: Key Mindsets - Truth-Seeking, Prioritisation, Moving Fast](https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking)

by [Neel Nanda](https://www.alignmentforum.org/users/neel-nanda-1?from=post_header)27th Apr 2025*This is post 2 of a sequence on my framework for doing and thinking about research.*[*Start here*](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK)*.*

Before I get into what exactly to do at each stage of the research process, it’s worth reflecting on the key mindsets that are crucial throughout the process, and how they should manifest at each stage.

I think the most important mindsets are:

- ***Truth-seeking***: By default, many research insights will be false - finding truth is hard. It’s not enough to just know this, **you must put in active effort to be skeptical and resist bias**, lest you risk your research being worthless.
- ***Prioritisation***: You have finite time, and a *lot*of possible actions. **Your project will live or die according to whether you pick good ones.**
- ***Moving fast***: You have finite time and a lot to do. This doesn’t just mean “push yourself to go faster” - **there’s a lot of ways to eliminate inefficiency without sacrificing quality**.
	- In particular, you must **learn to act without knowing the “correct” next step**, and avoid analysis paralysis.

**Warning**: It is extremely hard to be anywhere near perfect on one of these mindsets, let alone all three. I’m trying to describe an ideal worth aiming towards, but you should be realistic about the amount of mistakes you will make - I certainly am nowhere near the ideal on any of these! **Please interpret this post as a list of ideals to aim for, not something to beat yourself up about failing to meet.**

## Truth Seeking

Our ultimate goal in doing research is to uncover the truth about what’s really going on in the domain of interest. The truth exists, whether I like it or not, and being a good researcher is about understanding it regardless.

- This *sounds* pretty obvious. Who doesn't like truth? It’s easy to see this section, dismiss it as obvious and move on. But in practice this is extremely hard to achieve.
	- We have [many biases](https://en.wikipedia.org/wiki/List_of_cognitive_biases) that cut against finding truth
	- Insufficient skepticism doesn't *feel* like insufficient skepticism from the inside. It just feels like doing research.
- This means that **you must be putting in constant active effort into ensuring your results are robust**. This **must be integrated into part of your research process** - if you’re not, then there’s a good chance your results are BS.
	- “[Just try harder to be skeptical](https://www.neelnanda.io/blog/mini-blog-post-6-stop-pressing-the-try-harder-button)” is empirically a fairly ineffective strategy
	- One of the most common reasons I dismiss a paper is because I see a simple and boring explanation for the author’s observations, and they didn’t test for it - this often renders the results basically worthless.
		- I’d estimate that at least 50% of papers are basically useless due to insufficient skepticism

**What does putting in active effort actually mean**?

This takes different forms for the different stages:

- For exploration, the key failure mode is **not being creative enough when thinking about hypotheses**, getting attached to one or two ideas, and missing out on what’s actually going on.
	- Resist the urge to move on to the understanding stage the moment you have a plausible hypothesis - are there any unexplained anomalies? Could you do more experiments to gain more surface area first? What other hypotheses could explain your results? Etc
	- The standard hypothesis testing framework can be misleading here, because it has an implicit frame of being able to list all the hypotheses. But actually, **most of your probability mass should normally be on “something I haven’t thought of yet”**
		- You should regularly zoom out and look for alternative hypotheses for your observations. Asking another researcher, especially a mentor is a great source of perspective, asking LLMs is very cheap and can be effective.
		- That said, I still often find it helpful to think in a Bayesian way when doing research - if I have two hypotheses, how likely was some piece of evidence under each, and how should I update? Exploration often finds scattered pieces of inconclusive evidence, and there’s a skill to integrating them well.
	- **It’s not too bad if you end up believing false things for a bit**, the key thing is to move fast and reflexively try to falsify any beliefs you form, so you don’t get stuck in a rabbit hole based on false premises. This means it’s totally fine to investigate case studies and qualitative data, e.g. a deep dive into a single prompt.
		- If you’re getting lots of (diverse) information per unit time you’ll notice any issues.
	- **It is also an issue if you are*****too*****skeptical** and don’t let yourself explore the implications of promising but unproven hypotheses, as this is crucial to designing good experiments
- For understanding, you want to be careful and precise about **what your experiments*****actually*****show you**, **alternative explanations** for your results, whether your **experiments make sense on a conceptual level**, etc.
	- Here the Bayesian frame is often helpful. It’s generally overkill to put explicit numbers on everything, but it reminds me to ask the question “**was this observation*****more*****likely under hypothesis A or B**”, not just whether it was predicted by my favourite hypothesis
	- In exploration it’s OK to be somewhat qualitative and case study focused, but here you want to be more quantitative. If you must do **qualitative case studies**, do them on **randomly sampled things**, (or at least several examples, if your sampling space is small) )since it’s *so* easy to implicitly cherry-pick
		- The one exception is if your hypothesis is “there exists at least one example of phenomenon X”, e.g. ‘[we found multidimensional SAE latents](https://arxiv.org/abs/2405.14860)’.
- For distillation, in addition to the above, it’s important to **avoid the temptations of choosing a narrative that looks good**, rather than the best way to communicate the truth.
	- E.g. [**publishing**](https://arxiv.org/abs/2502.16681)[**negative**](https://arxiv.org/abs/2410.19278)[**results**](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
		- While it can be emotionally hard to acknowledge to *myself*that my results are negative, mechanistic interpretability has a healthy culture and **I’ve gotten nothing but positive feedback for publishing negative results**.
	- E.g. **exaggerating results** or stating an **overconfident narrative** to seem more publishable.
		- I find it pretty easy to tell when a paper is doing this - generally you should care more about impressing the more experienced researchers in a field, who are least likely to be fooled by this! So I don’t even think it’s a good selfish strategy.
	- E.g. **not acknowledging and discussing key limitations.**
		- If I notice a key limitation that a paper has not addressed or acknowledged, I think far less of the paper.
		- If a paper discusses limitations, and provides a nuanced partial rebuttal, I think well of it.

## Prioritisation

Ultimately, time is scarce. The space of possible actions you can take when doing research is wide and open ended, and some are far more valuable than others. **The difference between a failed and a great research project is often prioritisation skill.** Improved prioritisation is one of the key sources of value I add as a mentor

- Fundamentally, **good prioritisation is about having a clear goal (north star) in mind.**
	- You need **good judgement** about how well different actions achieve this goal
		- You need to **actually make the time** to think about how well actions achieve this goal!
	- You need to **be ruthless**about dropping less promising directions where necessary.
		- But **beware switching costs** - if you switch all the time without exploring anything properly you’ll learn nothing!
- The goals at each stage are:
	- *Ideation:***Choose a fruitful problem**
	- *Exploration*: **Gain information and surface area on the problem**
	- *Understanding*: **Find enough evidence to convince*****you*****of some key hypotheses**
	- *Distillation*: **Distill your research into concise, well-supported truth, and communicate this to the world.**
- Being great at prioritisation is pretty difficult, and requires good research taste, which will take a lot of time to develop. But **there’s often basic mistakes and low-hanging fruit to improve, if you just try.**
	- The first step is just making time to stop and ask yourself “**do I endorse what I’m doing, and could I be doing something better**?”
		- This advice may seem obvious, but is deceptively hard to put into practice! You need regular prompts  **Often it’s very easy to think of a better idea, but by default nothing prompts you to think.**
	- I like to **explicitly write goals down and regularly check in** that they’re being achieved - it sounds obvious, but you would be shocked at how effective it is to ask people if they’re doing the best thing for the project goals. I think in 3 tiers of goals:
		- Goal: What is the overall north star of the project? (generally measured in months)
		- Sub-goal: What is my current bit of the project working towards (measured in weeks)
		- Objective: What is the concrete short-term outcome I am aiming for right now (measured in days, e.g. 1 week)
	- I recommend **actually writing a plan**, and **estimate how long each step will take**, at least for the current research stage you’re in.
		- You don’t need to take it very seriously, and you’ll totally deviate a ton.
		- But **it forces you to think through the project**, notice uncertainties you could ask someone about, question if parts are really necessary to achieve your goals.
		- This is most important for understanding & distillation, though *can*be useful for exploration
		- **If you feel stuck,**[**set a 5 minute timer**](https://www.neelnanda.io/blog/post-28-on-creativity-the-joys-of-5-minute-timers) and brainstorm possible things you could do!
		- I typically wouldn’t spend more than a few hours on this
			- Unless you have a mentor giving high quality feedback - then it’s a great way to elicit their advice!
			- But even then, feel free to deviate - mentors typically have good research *priors*, but you know way more about your specific problem than them, which can be enough to make better decisions than even a very senior researcher
- **You need to prioritise at many different layers of abstraction**, from deciding when to move on from an experiment to deciding which hypothesis to test first to deciding when to give up on testing a hypothesis and pivot to something else (or just back to exploration)
- **Prioritising and executing are different mental modes and should not be done simultaneously**. Keep them separate, and make time to regularly reflect, and time to lock-in and execute on a plan without stressing about if it’s the best plan
	- Concrete advice: Work to a schedule where you regularly (ideally at least once a day, and with extended reflection at least once a week), zoom out and check that what you’re doing is your highest priority. E.g. work in pomodoros
	- **Having a**[**weekly review**](https://www.neelnanda.io/blog/39-reflection)**can be incredibly useful** -  where you zoom out and check in on what’s going on, any current issues, etc. Some useful prompts:
		- What is my goal right now?
		- What progress have I made towards that goal?
		- What’s consumed the most time recently?
		- What’s blocked me?
		- What mistakes have I made, and how could I systematically change my approach so it doesn’t happen again in future?
		- What am I currently confused about?
		- Am I missing something?
- See Jacob Steinhardt’s [excellent blog post on research prioritisation](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html).
- **Warning**: Different people need to hear different advice! (An eternal issue of writing public advice…). Some get stuck in rabbit holes and need to get better at moving on. Others get caught in analysis paralysis and never do *anything*, because they’re always waiting for the (non-existent) perfect opportunity.
	- **Real prioritisation is about a careful balance between exploration and exploitation**.
	- You probably know which failure mode you tend towards. **Please focus on the advice relevant to you, and ignore the rest**!

## Moving Fast

A core aspect of taking action in general is being able to move fast. Researchers vary a lot in their rate of productive output, and it gets very high in the best people - this is something I value a lot in potential hires.

This isn’t just about working long hours or cutting corners - there’s a lot of skill to **having fast feedback loops**, **noticing and fixing inefficiency** where appropriate, and **being able to take action or reflect where appropriate**. In some ways this is just another lens onto prioritisation.

- **Tight feedback loops are crucial**: A key thing to track when doing research is your feedback loops.
	- **Definition**: A **feedback loop** is the process from having an experiment idea and to results. Tight feedback loops are when the time taken is short.
	- It will make an enormous difference to your research velocity if you can get your feedback loops as tight as possible, and **this is a big priority**.
		- This is because you typically start a project confused, and **you need to repeatedly get feedback from reality to understand what’s going on**. This inherently requires a bunch of feedback loops that can’t be parallelised, so you want them to be as short as possible.
		- This is one of the big advantages of mech interp over other fields of ML - we can get much shorter feedback loops.
	- A mindset that I often find helpful is a deep-seated sense of impatience and **a feeling that something should be possible to do faster**. Sometimes I just need to accept that it will take a while, but often there is a better way, or at least a way that things can be reduced.
	- **Coding in a notebook is a lifesaver** (eg Jupyter, VS Code Interactive Mode or Colab)
	- Tips for tight feedback loops in mech interp:
		- Putting your data in a data frame rather than in a rigid plotting framework like Weights and Biases allows you to try arbitrary visualizations rapidly.
		- De-risking things on the smallest model you can, such as writing code and testing it on a small model before testing it on the model you're actually interested in.
		- Train things on fairly small amounts of data just to verify that you're seeing signs of life.
		- Sometimes there’s irreducible length, e.g. you need to train a model/SAE and this takes a while, but you can still often do something - train on less data, have evals that let you fail fast, etc.
- **Good tooling accelerates everything**. All stages benefit from **flexible exploration tools**(e.g., interactive notebooks, libraries like TransformerLens or nnsight), efficient infrastructure for running experiments, and helpful utilities (e.g., plotting functions, data loaders).
	- Flexible tooling tightens feedback loops by shortening the time between an arbitrary creative experiment idea and results, even if it’s less efficient for any given idea.
	- The balance shifts: more flexibility needed early, more optimization/robustness potentially useful later e.g. during the distillation stage it can make sense to write a library to really easily do a specific kind of fine-tuning run that happens a ton
- A corollary of this is that **you should (often) do fast experiments first**. It is far better to do a quick and dirty experiment to get some preliminary signs of life than an extremely long and expensive experiment that will produce conclusive data but only after weeks of work.
	- Realistically you should be prioritising by information gain per unit time.
	- This is especially important in exploration where it's hard to have a clear sense of which experiments are the most useful while estimating their tractability is pretty easy. When distilling you may know enough to be comfortable implementing a long running but conclusive experiment.
- **Audit your time**. It's all well and good to talk about the importance of speed and moving fast, but how do you actually do this in practice? One thing that might be helpful is to log how you spend your time and then reflect on it, and ways you might be able to go faster next time.
	- For example, you could use a tool like [Toggl](https://toggl.com/) to roughly track what you're doing each day and then look back on how long everything took you and ask, "**How could I have done this faster**? Was this a good use of my time?"
		- Often it’s easy to fix inefficiencies and the hard part is noticing them - e.g. making a util function for a common tedious task, or noticing things that an LLM could automate.
	- Note: It is *not*productive to look back and feel really guilty about wasting time. **Nobody is perfect and you will always waste time**. I am advocating for maintaining a mindset of **optimism that you will be able to do even better next time**.
- **Fail fast**. One of the largest time sinks possible is **investing weeks to months of effort into a failed research direction**. Thus, a key question to ask yourself is: if this direction is doomed, how could I discover this as fast as humanly possible?
	- I often try to think through what kind of confident predictions a hypothesis I care about makes in the understanding stage, or what fundamental assumptions make me think my domain is interesting at all in the exploration stage, and then think of the quickest and dirtiest experiments I can to test these.
		- It's often much better to have several quick and dirty experiments to attack different angles where you could fail fast than to put a lot of effort into one.
- **Are you moving*****too*****fast?**This is a natural pushback to the advice of ‘try hard to move fast’. It’s easy to e.g. be sloppy in the name of speed and introduce many bugs that cost you time in the long-run.
	- This is a hard balance, and I largely recommend just exploring and seeing how things go. But there *are*often things that can speed you up beyond ‘just push yourself to go harder in the moment’, which don’t have these trade-offs, like choosing the right experiments to run.
	- **Make sure you still regularly take time to think and reflect, rather than feeling pressure to constantly produce results**

### Taking action under uncertainty

A difficulty worth emphasising when trying to move fast is that there are a *lot*of possible next steps when doing research. And it’s pretty difficult to predict how they’ll go. Prioritisation remains crucial, but this means it’s also very hard, and **you will be highly uncertain about the best next step**. A crucial mindset is **being able to do something anyway, despite being so uncertain.**

- As a former pure mathematician, this is something I’ve struggled a fair bit with - I miss doing things grounded in pure, universal truth! But it’s learnable
- Ultimately, you just need to accept on an emotional level that you don’t get to know the “right” answer for what to do next - **in practice, there’s no such thing as the right answer**.
	- The ideal is to strive to carefully evaluate the extremely noisy evidence, make a best guess for what to do next, and act on it, while also being self-aware enough to notice if it no longer seems the best action. This is a hard balance to achieve, but super useful if you can do it.
- Especially when you’re starting out, this can be very low stakes: **the value of anything you do is dominated by the learning value**! If you make bad decisions you will learn and can do better next time, so it’s hard to really have a bad outcome.

# [My Research Process: Understanding and Cultivating Research Taste](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research)

by [Neel Nanda](https://www.alignmentforum.org/users/neel-nanda-1?from=post_header)2nd May 2025*This is post 3 of a sequence on my framework for doing and thinking about research.*[*Start here*](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK)*. Thanks to my co-author Gemini 2.5 Pro*

# Introduction

Spend enough time around researchers, and you'll hear talk of "research taste." It's often presented as a somewhat mystical quality distinguishing the seasoned research from the novice – an almost innate sense for which research ideas will flourish and which will fail. While I believe research taste is very real, incredibly valuable, and a key differentiator I look for, I *don't* think it's mystical or innate. Talent plays an important role, but taste is largely learned, and with the right mindset you can learn faster.

**What is research taste?**As I define it, research taste is far broader than just picking the right problem at the outset. Research is full of key decisions that will affect the future of the project, without an obvious way to find the right answer: from choosing the research problem itself, to identifying which anomalies are and are not worth exploring, distinguishing an experiment that will be compelling from one that’ll have inconclusive results, etc. I think of taste as **the set of intuitions and good judgment that guide a researcher’s decisions*****throughout*****the research process**, any time an ambiguous or open-ended decision like this arises. This can just be gut feeling, but also having conceptual frameworks you reason through, having novel ideas spark in your mind, etc.

**Where does taste come from?**If you're new to research, feeling like you lack "taste" is completely normal and expected. You don't need perfect judgment to start. In fact, trying to force it early on can be counterproductive. Think of training your intuition like training a network. It starts poorly initialized and needs lots of diverse, high-quality training data (i.e., research experience). With time, people often develop fairly deep and sophisticated taste, as they see enough examples of research outcomes, but this generally isn’t something people start with.

**How to learn it?**In my opinion, research taste is one of the hardest skills to learn for being a good researcher. To see why, let's lean more into this analogy of training a neural network. The core problem is **you just don't get that much data**. Generally the shorter a feedback loop is the more data you will get. By definition research taste is about things that are not immediately obvious. For designing a good experiment, sometimes you can get results from hours to day, but feedback on whether a research idea was good can take months!

I think the main way to speed it up is by **getting more data**, and by **being more sample efficient** about the data that you have. To get more data the easiest way is to **lean on sources of supervised data:**ideally**a mentor**, or **seeing what worked in papers**. You can also get more from each data point - analyse it in detail before setting the feedback, **predict your mentor’s answers before they give them**, etc. When you have made a research decision and you eventually get feedback, do a post-mortem analyzing what did and did not work and why and what general themes you could look at in future.

But even with all that, **expect learning taste to take a while**, especially high level strategic things like choosing a project - learning speed depends on your feedback loops, and taste has very slow ones. Further, research taste often translates poorly from other fields, or comes with counter-productive habits

# What is Taste?

As discussed, I define **research taste** broadly: **it's the collection of intuitions and judgments that guide good decision-making throughout a research project,** especially where feedback loops are long, and the search space is large and open-ended.

I take such a broad definition, because I think that the ability to make good judgements is a fairly general skill, and improving at one facet often helps you improve at all of them, by e.g. getting better conceptual frameworks and domain knowledge.

While **Problem Selection** (strategic judgment about tractability and interest) is the most visible aspect, research taste also covers:

- **Exploration:** A tactical sense for which experiments yield the most insight, recognizing interesting anomalies versus noise, knowing when to dig deeper or move on from a thread. *Does this surprising result feel like a key insight or a distracting artifact?*
	- My internal experience here looks like a visceral science of excitement vs boredom or flinching away from messiness/ugliess. I tend to get excited about things that feel like unexpected structure, spark follow-up experiments, or relate to a deep curiosity I have.
- **Understanding:** Designing creative, elegant experiments that cleanly distinguish hypotheses, judging the plausibility and explanatory power of different theories, identifying crucial assumptions or potential confounds. *Is this experiment truly isolating the variable I care about? What's the simplest explanation for this data?*
	- My internal experience is that I may have a beautiful hypothesis I *want*to believe, but it feels uncertain, and this creates an uncomfortable sense of instability.
	- I try to probe at where the instability comes from, what predictions are made by that potential flaw, and design an experiment to target it.
	- A good experiment design feels very clean and reliable - I would trust the results - while for a bad one I still have this shifting sense of uncertainty and being able to generate many alternative explanations
- **Communication & Distillation:** Identifying the core, communicable claims within messy findings, structuring a compelling and *true* narrative, anticipating audience confusion, knowing what makes a result impactful *to others*. *What's the single most important takeaway here? How can I present this evidence most clearly and honestly?*
	- My internal experience of compression is about having a frustration and impatience with length and unnecessary conceptual detail - I want to distill the research down into what is truly important, and reach a point where I can cut no further without sacrificing something important.
	- If I’ve compressed too far, there’s a sense that there’s a missed opportunity - a really exciting thread that’s missed out.

## Decomposing Research Taste

Where does this "taste" come from? In my experience, it boils down to a few key ingredients:

1. **Intuition (System 1):** This is the fast, gut-level feeling - what people normally think of when they say research taste. A sense of curiosity, excitement, boredom, or skepticism about a direction, experiment, or result.
	1. "This feels promising," "This feels like a rabbit hole," "This anomaly seems *important*," "This explanation feels too simple/too complex."
	2. This is the part that feels most like "taste" and develops slowly through repeated exposure and feedback - when I refer to gathering data to train a network, I largely mean training your intuition.
	3. Empirically, my own recommendations based on this intuition have a decent hit rate, and experienced researchers are often fantastic (though not flawless!) at this, but this takes time.
2. **Conceptual Framework (System 2):** This is deep domain knowledge and understanding of underlying principles.
	1. This is crucial in mech interp, especially as it’s a pre-paradigmatic field, where you can’t just memorise and apply a standard method.
		1. I’d guess it’s still important in other domains, though I am less sure
	2. For mech interp, this includes:
		1. Understanding transformer mechanics and basic facts - they’re autoregressive, the residual stream is the central object, tokens are discrete while all activations are continuous vectors, etc
		2. Key results and heuristics: like superposition or the linear representation hypothesis, or the idea that features and circuits exist at all
		3. Common techniques and where to use them and what they can tell you: patching, SAEs, probing, prompting, etc.
			1. This can get pretty deep! See [my paper on how to think about activation patching](https://arxiv.org/abs/2404.15255).
		4. Foundational knowledge of relevant adjacent fields: linear algebra, ML theory, training ML models, basic software engineering, etc
	3. This conceptual framework allows you to generate hypotheses, evaluate plausibility *explicitly*, spot inconsistencies, design sensible experiments, and explain *why* your intuition feels a certain way. It provides the structured reasoning to back up or override gut feelings.
	4. Eventually, this conceptual framework should feel like [a gears-level model](https://www.alignmentforum.org/w/gears-level), where you can reason about the key moving parts, and what would make a project or experiment idea work vs fail vs be impractical.
3. **Strategic Big Picture:** Understanding the broader context of the field. What problems are important? What are the major open questions? What approaches have been tried? What constitutes a novel contribution?
	1. My motivations for doing mech interp partly stem from making AGI safe, so the main big picture is “what work translates into better outcomes for AGI”, and being able to break this down into near-term steps.
	2. But even for less goal directed fields, where the goal is just curiosity driven basic science, there’s often a useful big picture around what advances would unlock many future advances or be a dead end, what would people care about, etc.
	3. Ideally, you dwell on the big picture enough that your intuitive sense of curiosity and excitement starts to integrate it - it’s not about overriding your curiosity with strategic obligations, it’s about aligning them so you’re excited about what matters. I see this as one input to prioritisation, among many.
4. **Conviction & Confidence:** Research inevitably involves setbacks. A certain level of conviction – a belief in the direction, resilience to negative results – is often instrumentally useful for perseverance. It helps you push through the messy exploration phase or refine an idea that isn't working perfectly yet.
	1. Empirically, research taste also often leads to conviction - the intuitive feeling that an idea is exciting and important tends to also give motivation and focus.
	2. However, this is a **double-edged sword**. Your intuitions are not well calibrated. **Confidence doesn't mean correctness**. Generally people reach the level of having conviction far before they reach the level of having correct intuitions
	3. **The ideal is*****strategic*****conviction**: the ability to adopt a confident mindset to maintain momentum, while regularly zooming out to reflect and maintaining the capacity for zoomed-out skepticism and the willingness to update or abandon course based on evidence.
	4. **Track data**: Conviction is instrumentally useful, but so are correct beliefs. Generally, the best way to get calibrated is to pursue an exciting idea and see it fail in unexpected ways. Try to **write down prior predictions**, and *why*you think an idea is good, pursue it, and **reflect on what happened**.
		1. Corollary: **It’s fine to be uncalibrated at first**, this can help you get more research done and gather more data, if you’re paying attention you’ll often get over it.
			1. I often mentor people who start out by getting way too attached to flawed ideas, and don’t engage well with criticism. Seeing some of their exciting ideas fail tends to helps a lot.

**These components interact**. A strong conceptual framework sharpens intuition. Experience builds both intuition and framework knowledge. Strategic awareness helps channel conviction productively.

# Cultivating Research Taste

If taste is like an ML model, how can we speed up training? We want to improve the quantity (and quality) of data, and the sample efficiency of how much we learn from it.

- **Learning more from each data point**: You will learn something just from doing research. You'll get some feedback, some experience, and your intuitions and models will improve. But each data point is actually much richer than just a binary of success or failure!
	- My recommendation is to **make explicit predictions**, **review accuracy**, and make time to **reflect on what you missed** and how you could do better next time.
		- Keep a research log. Ask *why* things worked or failed. Was it luck, execution, or a fundamental judgment call (taste)?
	- **Reflect Deliberately:** After an experiment or project phase, ask: What worked? What didn't? What surprised me? What would I do differently next time? How does this update my model of this domain? ([Weekly reviews](https://www.neelnanda.io/blog/39-reflection) can be great for this).
- **Getting more data**: The obvious source of data is doing research. But there are other sources too!
	- **Leverage Mentors:** This is perhaps the biggest accelerator. A mentor provides high-quality, curated "labels", insights and feedback. You can think of this as supervised data, in contrast to the slow RL of doing research yourself.
		- **Predict their advice:** Before asking your mentor ("Should I run experiment A or B?", "Is this result interesting?"), predict their answer and reasoning.
		- **Analyze surprises:** When their answer differs from your prediction, *dig into why*. What perspective, heuristic, or piece of knowledge did they use that you lacked? This is incredibly valuable training data for your internal model.
			- **Strong recommendation**: Do this by **repeatedly paraphrasing their reasoning**. Try to repeat back their arguments in your own words, and ask what you’re missing. This is an excellent way to ensure you’ve processed correctly, and often highlights misunderstandings. This is one of my most effective tactics when learning from people.
		- **Absorb their frameworks:** Listen not just to *what* they advise, but *how* they reason. What questions do they ask? What principles do they seem to operate by?
	- **Learn Critically from Papers (Offline Data):** Papers are a biased dataset (publication bias!), but still useful.
		- Read actively: Predict methods, results, and limitations before revealing them.
		- Ask *why*: Why did the authors make these choices? What alternative approaches might they have considered? What makes this paper impactful (or not)?
		- Focus on *reasoning*: Try to reconstruct the authors' thought process, not just memorize the outcome.
		- Note: **Papers are*****very*****often flawed**! A common mistake in new researchers is assuming that everything in a paper was reasonable or done for principled reasons. Even in great papers, there’s a lot of janky crap or flaws in there. And many papers are just inherently flawed or outright false. Critically engaging with a paper’s flaws is also very educational
	- **Collaborate and Discuss:** Talk to peers. Explain your research plans and reasoning. Listen to theirs. Critique each other's logic. Explaining forces clarity and exposes flawed assumptions. Hearing others' perspectives provides diverse 'data points'.
	- **Prioritize Projects with Clearer Feedback:** Especially early on, projects where you can test intermediate hypotheses or get partial results relatively quickly can accelerate learning more than moonshots with year-long feedback loops.
- **Feedback loops**: The speed at which you complete each loop for each facet of taste determines how fast you learn that aspect.
	- **Short Loops/tactical taste:** Designing a specific experiment, debugging code, interpreting a single plot. Feedback is often quick (minutes to days). You'll likely improve *much*faster at skills with short feedback loops.
	- **Long Loops/strategic taste:** Choosing a research problem, deciding on a major strategic direction. Feedback might take months or even years. **Improvement here is inherently slower.**
	- **Implication:** Don't beat yourself up if your high-level strategic taste develops slower than your tactical experimental skills. This is expected.

I have less to say about other components of research taste like conceptual understanding or strategic picture - generally a similar mindset works there, though as it’s no longer really a black box I think it’s more straightforward, and is much easier to learn from reading papers and existing resources, and talking to mentors/experts. Conviction is more of a matter of personality and preference, in my experience.

# Conclusion: Patience and Process

Research taste isn't magic. It's a complex set of intuitions and frameworks built incrementally through experience, reflection, and learning from others. It governs the crucial, often implicit, decisions that shape a research project's success.

Because the feedback loops for high-level strategic taste are long and noisy, don't expect to master it quickly. It's perfectly normal, and indeed expected, to rely heavily on external guidance (like mentors or established research directions) early in your career. Focus first on mastering the skills with shorter feedback loops – coding, running experiments, analyzing data, clearly communicating simple results.

By actively engaging in research, deliberately reflecting on your decisions and their outcomes, and strategically leveraging the experiences of others, you can accelerate the development of your own research taste. Be patient with the process, especially the long-game aspects like problem selection. Trust that by doing the work and learning effectively from it, your intuition will improve over time.

*Post 4, on ideation/choosing a research problem, is coming out soon - if you’re impatient you can read a draft of the whole sequence*[*here*](https://docs.google.com/document/d/1YMkeMrhqsWxZcNDD9CIUWEK_DAOegeufnbc79U2hycg/edit?tab=t.0)*.*

# Research as a Stochastic Decision Process
*by Jacob Steinhardt*

In this post I will talk about an approach to research (and other projects that involve high uncertainty) that has substantially improved my productivity. Before implementing this approach, I made little research progress for over a year; afterwards, I completed one project every four months on average. Other changes also contributed, but I expect the ideas here to at least double your productivity if you aren't already employing a similar process.

Below I analyze how to approach a project that has many somewhat independent sources of uncertainty (we can often think of these as multiple "steps" or "parts" that each have some probability of success). Is it best to do these steps from easiest to hardest? From hardest to easiest? From quickest to slowest? We will eventually see that a good principle is to "reduce uncertainty at the fastest possible rate". After revealing issues with more simplistic approaches, I will articulate this principle in detail and show how to apply it. Throughout, I draw my examples primarily from problems in machine learning and mathematics, but I believe that the principles generalize to other situations as well.

### Warm-Up

Suppose you are embarking on a project with several parts, all of which must succeed for the project to succeed. For instance, a proof strategy might rely on proving several intermediate results, or an applied project might require achieving high enough speed and accuracy on several components. What is a good strategy for approaching such a project? For me, the most intuitively appealing strategy is something like the following:

**(Naive Strategy)**
Complete the components in increasing order of difficulty, from easiest to hardest.

This is psychologically tempting: you do what you know how to do first, which can provide a good warm-up to the harder parts of the project. This used to be my default strategy, but often the following happened: I would do all the easy parts, then get to the hard part and encounter a fundamental obstacle that required scrapping the entire plan and coming up with a new one. For instance, I might spend a while wrestling with a certain algorithm to make sure it had the statistical consistency properties I wanted, but then realize that the algorithm was not flexible enough to handle realistic use cases.

The work on the easy parts was mostly wasted--it wasn't that I could replace the hard part with a different hard part; rather, I needed to re-think the entire structure, which included throwing away the "progress" from solving the easy parts.

What might be a better strategy than the naive strategy above? Since the naive strategy has the problem that we waste effort on the easy components if the hard components are intractable, maybe it would be better to complete the components in *decreasing* order of difficulty, starting from the hardest and moving to the easiest.

This *might* be better, but our intuitive sense of hardness likely combines many factors--the likelihood that the task fails, the time it takes to complete, and perhaps others as well. Here is an example:

Task A is a detailed and tricky calculation, but you have done many similar calculations before and are confident that given a few days you will succeed. Task B will likely take much less time, but it is something you haven't done before (so it is more likely there will be an unforeseen difficulty or problem).

In this case, task B would be better to do first--if you do task A first and then B turns out doomed, you have wasted several days. Even if A also has some chance of failing (so that it is both more likely to fail and takes longer than B), we would still usually rather do B before A.

This reveals that harder tasks should not necessarily be prioritized. Rather, we should prioritize tasks that *are more likely to fail*(so that we remove the risk of them failing) but also tasks that *take less time* (so that we've wasted less time if one of the tasks does fail, and also so that we get information about tasks more quickly).

### A Better Strategy: Sorting by Information Rate

We can incorporate both of the above desiderata by sorting the tasks based on which are *most informative per unit time*.

**(Better Strategy)**
Do the components in order from most informative per unit time to least informative per unit time.

To implement this, we need a method for quantifying informativeness. I will present two methods below--one based on *expected time saved*, and one based on *failure rate*. Rather than define these rigorously upfront, I will work through several examples, which should make the general case evident.

**Method 1: Expected Time Saved**

If an earlier step fails, we save time by not having to attempt the later steps. We should therefore complete the steps in the order that maximizes the expected value of the time that we save. We assume for now that we can actually quantify the probability that each step succeeds, as well as the time it will take. Consider the following example:

Example 1: All of the steps of a project have roughly equal chance of success (80%, say) but take varying amounts of time to complete.

In this example we would want to do the quickest task first and slowest last, since the later a task occurs, the more likely we will get to skip doing it. Sorting "easiest to hardest" is therefore correct here, but it is rare that all steps have equal success probability.

Example 2: An easy task has a 90% success probability and takes 30 minutes, and a hard task has a 40% success probability and takes 4 hours.

Here we should do the easy task first: if it fails we save 240 minutes, so 0.1 \* 240 = 24 minutes in expectation; conversely if the hard task is done first and fails, we save 30 minutes, for 0.6 \* 30 = 18 minutes in expectation. But if the hard task takes 2 hours or the easy task has a 95% chance of success, we should do the hard task first.

Thus, in this method we formalized "most informative per unit time" by looking at how much time we save (in expectation) by not having to do the tasks that occur after the first failure. Our computations assumed that we only find out if a task succeeds or fails at the end, as opposed to in the middle; however, they can be modified to take such complications into account.

For more than two tasks, this calculation method quickly becomes intractable: for K tasks we have to consider all K! permutations to find the best one. The next method avoids this issue.

**Method 2: Failure Rate**

This next method models the occurrence of failures as a Poisson process: if a task takes 30 minutes and has a 15% chance of failure, then there is about a 0.5% chance that the failure will occur in each minute (actually, it is slightly more than that because of overlap among the failures; the actual value is the solution p to (1-p)^30 = 0.85). Note that this differs from our previous assumption that failures can only occur at the end. This alternate model will simplify our calculations.

Formally, assume that the probability that we realize the task fails in the next minute is independent of how long we have been doing the task. Then the occurrence of a failure is a [Poisson arrival process](https://en.wikipedia.org/wiki/Poisson_point_process#Interpreted_as_a_point_process_on_the_real_line) and the time at which a failure occurs [follows an exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution#Applications_of_exponential_distribution) with some rate parameter ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") , where ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") tells us how frequently failures occur per unit time. Using basic properties of Poisson processes (see Appendix A), we can compute ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") as

![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as7.png "image_tooltip") ,

where ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as9.png "image_tooltip") is the success probability of the task.

This rate exactly tells us how quickly we will encounter failures while doing a given task. Since we would like to front-load failures as much as possible, we would always like to sort the tasks in decreasing order of their rate ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") .

Returning to Example 2, we can compute the rate ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") for the two tasks:

Task 1: ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as1.png "image_tooltip")

Task 2: ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as14.png "image_tooltip")

This new computation reverses our previous conclusion: The hard task has a higher rate, so is actually (slightly) better to do first! The reason for this is that the Poisson assumption implies that the higher the failure probability of a task, the faster (in expectation) we will encounter the failure. This contrasts with the previous assumption that we only encounter failures at the end of a task. We should keep in mind that both of these assumptions are likely somewhat incorrect in practice.

The rate method extends easily to more than two tasks, since we can simply sort tasks in order of ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") .

**An Additional Example**

In the case of the time-consuming but certain task A and quicker but uncertain task B, task A might take 12 hours but have a 90% chance of success, while task B takes 2 hours but has a 65% chance of success.

First, let's see what we get using the time saved method:

- A first: 0.1 \* 2 = 0.2 hours
- B first: 0.35 \* 12 = 4.2 hours

Now suppose we use the rate method:

- A first: log(1/0.9)/12 = 0.009
- B first: log(1/0.65)/2 = 0.215

B dominates A on *both* failure prob and time, so doing B first looks substantially better under both methods.

**Caveats**

These numbers are all completely made up and in practice you won't be able to estimate things so well. I subjectively distinguish between different "buckets" of success probability, such as:

- "I am confident that this can be done and that there are no unforeseen difficulties" (~95%)
- "I am confident that this can be done modulo Murphy's law" (~90%)
- "I see the basic path to accomplishing this and all the steps seem like they should work" (~65%)
- "I have the intuition that this should be possible but only have a murky view of the path" (~30%)

On the other hand, I tend to have much better estimates of task completion times if I've been practicing (~30% average relative error, albeit with large tails). You can get better at this within a few weeks by estimating completion times for each task and then recording the actual completion times in a daily log. You should also practice decomposing tasks into small actionable chunks, each taking roughly 20 minutes to 2 hours.

### A further improvement: opening up the "task" black box

Sorting tasks in decreasing order of failure rate is a good start; it should improve efficiency by a factor of 2-3. However, we can do *much* better still by learning to front-load the information gained about each task. Front-loading information requires a mental finesse: rather than seeking to complete a task, we must seek information *about* a task.

Specifically, for each task we want a *cheap way to obtain high confidence about whether that task will be feasible*. This is called "de-risking". The following pattern is indispensable:

**(Basic Pattern)**
De-risk all components (to the extent feasible), then execute.

As an example, suppose we wish to set up a dataset and then train a suitable model on that dataset. However, setting up the dataset is arduous: we must download it to a place with enough disc space, parse it into a usable format, and incorporate auxiliary data sources (like noun/verb banks for natural language processing).

Setting up the dataset and training the model are both time-consuming and either one could fail. Even worse, it would seem that we are forced to set up the dataset first, even though it is probably the more time-consuming task.

To avoid this issue, we could first download a few thousand examples. We can then examine several examples by hand, as well as compute some aggregate statistics, to assess whether the dataset has the properties we want. Ideally, this will reduce a lot of uncertainty about whether the will dataset is suitable.[1](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html#fn1)

### General principle: stochastic decision process

We can unify and extend the above insights by modeling a research project as a *stochastic decision process*. Specifically, we think of research as a multi-round game, where in each round we take some action that gives us some information; the information we get is stochastic, and well as perhaps the time needed to complete the action. We have two competing goals:

- Maximize probability of eventual success (don't give up if it turns out we can eventually solve the problem).
- Minimize expected time spent (give up early if the problem is not feasible, and solve the problem quickly if it is feasible).

We are often either in "de-risking mode" (determining if the problem is infeasible as quickly as possible) or "execution mode" (assuming the problem is feasible and trying to solve it quickly).

**An aside: tooling.** This picture grows more complicated if we consider actions that could speed up a family of future actions (such as writing helpful scripts to automate tasks, or reducing the execution time of the system). Such "tooling" tasks are tricky to model, because it seems we should implement tooling as soon as we know we will eventually want it (since it speeds up things that come after it). However, this ignores that more experience often yields refined desiderata for the tools we implement. There is thus a trade-off between building tools earlier vs. building better-targeted tools. I won't say more about this here, but it is an important point to keep in mind.

Another complication is that our ultimate goal is often nebulous--we are not asking "is this problem possible" so much as "how interesting of a problem in this space is it feasible to solve"? But I don't think this substantially alters the above principles.

### Some further practical ideas

There are a number of useful patterns for putting the above principles into practice. I list several below.

**For empirical work, measuring "ceilings" (an upper bound of how high performance could possibly be) is often useful.**Example: suppose we wish to build a system with 3 components that interact in a complicated way. One of the components is difficult to implement, but we can easily substitute a "cheating" version of that component (e.g. by looking at the test set or by using information that won't be available at deployment time). We often benefit by building a prototype system that initially uses this cheating version:

- If the system works, we know that a sufficiently good implementation of the difficult component will yield a working system.
- If the system doesn't work, we've saved the time of implementing the difficult component.

We can choose which components to cheat on initially, and which to implement fully, using the "informativeness per unit time" heuristic from above. For instance, if the ability to do well on a specific component is the major source of uncertainty in the project, cheating on it might be counterproductive (we may instead want to cheat on *everything but that component*).

The counterpart to ceilings are *baselines*--simple or off-the-shelf methods that give a quick lower bound on achievable accuracy. Baselines provide an important sanity check, as complicated methods often underperform simple baselines. Together with ceilings, they delineate a range of possible performance, which helps us interpret our core results.

**Brute force.**If we know of an easy-to-implement brute force solution and a difficult-to-implement fast solution, starting with the brute force solution has many of the same advantages as using ceilings, as long as the slower running time doesn't bottleneck prototyping. A brute force implementation also facilitates debugging the fast solution, since we can compare the outputs of the two algorithms.

As with ceilings, brute force is most useful when implementing the fast solution is not a major source of uncertainty (e.g. it is routine but annoying, or is one of many sources of uncertainty).

**For theoretical work, looking for counterexamples is useful.**The simplest example of this: if we find a counterexample to the main result we want to prove, then we need to either give up or make stronger assumptions.

A more nuanced (and more common) example: if we are trying to prove that ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as6.png "image_tooltip") , and our current technique does this by proving ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as10.png "image_tooltip") and then ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as11.png "image_tooltip") , finding a counterexample to ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as10.png "image_tooltip") will rule out that technique.

Yet more nuanced/common: if we are trying to prove that ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as6.png "image_tooltip") , and our current technique applies equally well under assumptions ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as12.png "image_tooltip") and ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as0.png "image_tooltip") , then a counterexample to ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as13.png "image_tooltip") will rule out the technique.

**More generally, thinking about simplified instances of a problem is often useful.** This is because it provides intuition that often suggests/rules out approaches for the original problem. Similarly to de-risking, the ability to rule out entire approaches makes this tactic invaluable from the stochastic decision process perspective.

**Running simulations.**If we wish to prove X, first run simulations to check if X is actually true. This is easy when assessing the behavior of a specific algorithm, as we can simply run the algorithm. Simulations can also, for instance, help reveal the asymptotics of a random process, or be used to search for small counterexamples to a conjecture.

### Exponentially branching search trees

Another important mental framework focuses on the combinatorial aspect of a decision process:

**(Research as branching search)**
I often think about possible approaches to a problem as an exponentially branching search tree: we could try X, X', or X''. Then X could be combined with Y, Y', Y'', or X' could be combined with Z or Z', etc. This exponential blow-up poses barriers to projects with more than a small number of steps unless we have a way to systematically rule out entire branches of the tree.

Exponential branching often occurs because there are many ways to try a particular approach--perhaps we want to bound the moment generating function, and there are many ways to attempt this; or we think data augmentation will help our model generalize, but there are many ways to augment the data. With many possibilities for each step, even a two- or three-step approach creates a huge search space. For instance, if there are 10 ways to try bounding the moment generating function, and two other similar steps, then we have to try 1000 possibilities.[2](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html#fn2)

If the steps *factor*--meaning they can each be solved in isolation--this might be fine (we only have to try 3\*10 instead of 10^3 possibilities). However, I usually find that there is some interdependency between different steps. For a math problem, maybe how good of a bound I get from step 1 affects how hard I need to work for step 2. Or for an experiment, if any of 3 parts of the setup are wrong then the method just won't work, so I don't get signal until I've gotten a few things right simultaneously.

For this reason, I think it's *much* more useful to prune branches of the search tree at the level of conceptual approaches ("can the moment generating function give me sufficient control over the distribution I care about?") than at the level of a specific instantiation ("does this particular moment generating function bound work?"). This leads to adopting several principles:

**Whenever something doesn't work, I ask *why* it didn't work.** My goal is to avoid trying similar things that will fail for the same reason (or to notice that the reason why it didn't work is circumventable, and that a modified approach actually will work).

**Trying an experiment and seeing it fail gives little information by itself.** When an experiment fails, it is tempting to conclude "I tried X and it didn't work". However, if X is a high-level conceptual approach, then a more correct conclusion is "I tried an implementation comprising 0.1% of the possible implementations of X, and observed that that particular implementation did not work". For this reason, I am far less in favor than most people of publishing negative results, unless the negative result comes with insight into what caused the failure. In contrast to common concerns, negative results that come with such insights are [already publishable](https://acl2018.org/paper/1604/).

**Compared to other people I know, I try harder and earlier to show that my ideas can't work to solve a problem.** Importantly, it is often not obvious that multiple approaches to a problem all have the same issue. In the past, I have spent months trying different approaches to a problem before finally stepping back and realizing that they were all failing for the same reason. Moreover, I had all the data necessary to make this realization a couple weeks in but had failed to do so. I now save considerable time by ruling out ideas early on, and as a result I am usually bottlenecked on coming up with ideas rather than on implementing ideas.

**Additional Discussion**

In the previous section I talked about ruling out ideas. When ruling out ideas, it is important to hold oneself to a high standard. "This doesn't seem like it will work" or "I feel less motivated after trying a few things along this line that didn't work" are *not* ruling out an idea. We could perhaps think of them as updating the probabilities that a solution lies within a given subtree of the search tree. But these updates are rarely large updates, and I find them much less reliable than a solid argument for why an approach is doomed.

Note that I am *not* advocating that you should never trust your feelings. If you feel pessimistic about an approach, that is a great reason to try to show that the approach can't work! If I feel pessimistic about an approach but fail to rule out that it could work, I often then feel more optimistic.

I am also *not* advocating for the failure mode of only trying low-variance ideas, or of avoiding projects that lack an obviously promising approach. Part of the point of being able to systematically rule out ideas is to enable trying ideas that only have a low probability of working, or that do not immediately yield progress.

### Summary

Many of our default intuitions about how to pursue uncertain ideas are counterproductive:

- We often try easier tasks first, when instead we should try the most informative tasks first.
- We often conflate a high-level approach with a low-level instantiation of the approach.
- We are often too slow to try to disprove our own ideas.

Building frameworks that reify the research process as a concrete search problem can help unearth these incorrect intuitions and replace them with systematic reasoning.

### Appendix A: Poisson Process Calculation

In a Poisson process with rate ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") , the probability that a failure has already occurred by time t is ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as5.png "image_tooltip") , so in particular ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as8.png "image_tooltip") , where ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as4.png "image_tooltip") is the time to complete the task and ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as9.png "image_tooltip") is the success probability of the task. If we solve for this, we get that the rate ![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as3.png "image_tooltip") is equal to

![alt_text](https://cs.stanford.edu/~jsteinhardt/images/Research-as2.png "image_tooltip") ,

as claimed.

## Notes

---

1. This doesn't quite fit into the framework because if the dataset is unsuitable we can try again until we find a suitable dataset. But it could be that we try 4 datasets, they are all unsuitable, and we eventually conclude that there aren't any suitable datasets. The sort of de-risking above allows us to reach this conclusion much faster and avoid spending time trying to train a model on a broken dataset. [↩](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html#fnref1)
2. This is purely illustrative and in reality we can't necessarily decompose different attempts into a fixed number of discrete "ways" of attempting something. [↩](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html#fnref2)


# [Highly Opinionated Advice on How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)

by [Neel Nanda](https://www.alignmentforum.org/users/neel-nanda-1?from=post_header)12th May 2025## TL;DR

- **The essence of an ideal paper** is the **narrative**: a short, rigorous and evidence-based technical story you tell, with a takeaway the readers care about
	- **What?**A narrative is fundamentally about a contribution to our body of knowledge: **one to three specific novel claims** that fit within a cohesive theme
	- **Why?**You need **rigorous empirical evidence**that convincingly supports your claims
	- **So what?**Why should the reader care?
		- What is the **motivation**, the problem you’re trying to solve, the way it all fits in the bigger picture?
		- What is the **impact**? Why does your takeaway matter? The **north star** of a paper is ensuring the reader **understands**and **remembers**the narrative, and **believes** that the paper’s evidence supports it
- The first step is to **compress your research** into these claims.
- The paper must **clearly motivate these claims, explain them on an intuitive and technical level**, and **contextualise what’s novel**in terms of the prior literature
	- This is the role of the abstract & introduction
- **Experimental Evidence**: This is absolutely crucial to get right and aggressively red-team, it’s how you resist the temptation of elegant but false narratives.
	- **Quality > Quantity**: find compelling experiments, not a ton of vaguely relevant ones.
	- **The experiments and results must be explained in full technical detail** - start high-level in the intro/abstract, show results in figures, and get increasingly detailed in the main body and appendix.
		- **Ensure researchers can check your work**- provide sufficient detail to be replicated
		- **Define key terms and techniques** - readers have less context than you think.
- **Write iteratively**: Write abstract -> bullet point outline -> introduction -> first full draft -> repeat
	- Get feedback and reflect after each stage
	- Spend comparable amounts of time on each of: the abstract, the intro, the figures, and everything else - they have about the same number\_of\_readers \* time\_to\_read
- **Inform, not persuade**: Avoid the trap of overclaiming or ignoring limitations. Scientific integrity may get you less hype, but gains respect from the researchers who matter.
- **Precision, not obfuscation**: Use jargon where needed to precisely state your point, but not for the sake of sounding smart. Use simple language wherever possible.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/081539a4d3b61418cfc6657e29bac2d234e1adf5331b4a7b3ea7974ad4d2bfc1/pact399wy3qd43yhrzx8)

***Case study**: The abstract of*[*refusal is mediated by a single direction*](https://arxiv.org/abs/2406.11717?)*, broken down into the purpose of each sentence*

## Introduction

**Your research only matters if people read, understand, and build upon it**. This means that **writing a good paper is a critical part of the research process**. Further, the process of writing forces you to clarify your own thinking in ways that often reveal gaps or new insights - I’ve often only properly understood an idea after writing it up. Yet, to many, writing feels less fun than research and is treated as an after thought - **a common but critical mistake**.

In my experience supervising 20+ papers and reading/appreciating/being annoyed by a bunch more, I've developed my own opinionated framework for what I think makes a paper good and how to approach the writing process. I try to lay this out in this post, along with a bunch of concrete advice.[[1]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnwwk8u16jdjf) This post assumes you’ve already done a bunch of technical research, and focuses on how to effectively share it with the world, [see my other posts](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand) for advice on the research part.

**Caveat**: I mostly have experience with writing mechanistic interpretability papers and this advice is written with that flavour. I expect much of it to generalise to the rest of ML and some to generalise to other fields but it's hard for me to say. Further, this is very much my personal opinionated, and optimised more for truth-seeking than getting into conferences[[2]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fn5kbv6xekx6j). See other great advice [here](https://www.jakobfoerster.com/how-to-ml-paper) and [here](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html).

**Caveat 2**: There are many reasonable objections to the academic paper as the format for communicating research. Alas, engaging with those is outside the scope of this post, it’s long enough as it is.

## The Essence of a Paper

At its core, a paper should **present a narrative** of **one to three specific concrete claims** that you believe to be true, that **build to some useful takeaway**(s). Everything else in the paper exists to support this narrative. The second pillar of the paper is **rigorous evidence for why they are true**- obviously there will always be some chance you’re wrong, but they should be compelling and believable, without obvious glaring flaws.

1. **Communicate the key idea**
	1. **Motivate** why someone should care about them
	2. **Contextualize** them in existing literature
2. **Communicate them precisely**, with all relevant technical detail, terminology and background context
3. **Provide sufficient evidence** to support them

### Crafting a Narrative

One of the critical steps that can make or break a paper is crafting a narrative. What does this actually mean? And how can you do it?

Research is about discovering new things and pushing forward our frontier of existing knowledge. I view a paper as something that finds insight and provides compelling evidence behind it. The way to tell when you could start writing a paper is when you have **learned something insightful**, in a way that could be **made legible to someone else**.

**This is far easier said than done**. A research project is often a mess of fun results, confusions, insights, and remaining mysteries. Even when you’ve made enough progress to write it up, you will likely have a great deal of tacit knowledge, interesting rabbit holes, dangling threads, etc - **projects rarely feel done**.

I find that converting a project into a great narrative is **a subtle and difficult skill**, one of the many facets of [**research taste**](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research). It’s something I’ve gotten much, much better at over time, and it’s hard to say what the best way to get better at it is, beyond experience. One exercise I’d recommend is taking papers you know, and trying to write down what their narrative is, and ask yourself what its strengths and weaknesses are. If at all possible, consult mentors/more experienced researchers for advice. But if you need to come up with one yourself, the right questions to ask look like:

- Which of these results would be most exciting to show someone?
- Actually show someone your findings and ask what they're most interested in
- What seems particularly important?
- Why should anyone care about this work?
- What was hard about what you did, that perhaps no one else has done?

A good, compelling narrative comes with motivation and impact. The key points to be sure to cover:

- The context of your insight
- The problem you're trying to solve
- Why this matters
- What you have shown
- Why the reader should believe it
- What the insight is

Why do you need this kind of compressed narrative? Often there’s far more insight in a research project than can be contained in this structure. But it is impossible to convey this level of nuance in a paper. Readers will rarely take away more than a few sentences of content. **Choose those sentences carefully**. These are the insights shared by your paper, **your contribution to the literature**. These are the specific, concrete claims that you want to communicate - you cannot reliably communicate much more. You will need to compress your research findings down into a handful of claims, prioritise those, and accept that you may need to drop a bunch of other detail, or move it to appendices. If you don’t deliberately de-prioritise some details, then something else will get dropped, which may have been far more important.

What do I mean by claims? For example:

- "Method X is the best approach on task Y (according to metric Z)"
- "A substantial part of the model's behavior in scenario A is explained by simple explanation B"
- "Technique C can fail in scenario D if conditions E and F hold"

One important claim, with sufficiently strong evidence, can be enough for a great paper! If you want multiple claims, I strongly recommend **choosing claims that fit together in a cohesive theme** - papers are far easier to understand, praise, share, etc if there is a **coherent narrative**, not just a grab-bag of unconnected ideas.

Depending on the strength of the evidence, you can adjust the confidence of a claim:

- **Existence-proof claims**: "We found at least one example where X happens" (like the indirect object identification paper providing an existence-proof for self-repair)
- **Systematic claims**: "X generally happens across a wide range of contexts" or "X is common"
- **Hedged claims**: “There is compelling/suggestive/tentative evidence that X is true”
- **Narrow claims**: “X is the best method for specific situations V & W, if your goal is objective Y”
- **Guarantees**: “X is always true”[[3]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnxjcznpnz657)

Generally, stronger statements make for more interesting papers, but require higher standards of evidence - resist the temptation to overclaim for clicks!

### When to Start?

Another thorny question is: When should you stop doing research and start writing up your research? This is a hard and subtle question that is, in many ways, a matter of [research taste](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research), but here is my general guide:

1. Write down a list of things you've learned
2. Review that list carefully, and ideally show it to someone else
3. Ask yourself how comfortable you would be defending the claim that you have provided meaningful, positive evidence for these results
4. Think about reasons why others might care about this
5. Focus on things you've done that have been hard or non-trivial and look for exciting elements

But generally, this is unfortunately just a hard thing to tell when starting out, and gets far easier with time and experience. If you can consult a more experienced researcher, definitely do.

A more meta piece of advice when starting out is to try to choose projects where the narrative will be pretty obvious, e.g. method X beats SOTA method Y in domain Z on metric W

**Warning**: Before moving into paper-writing mode, **it's crucial to verify that your evidence is actually correct**. An unfortunate fact is that **many published papers are basically false or wildly misleading**. Don't let this happen to you! Carefully check your critical experiments and, if possible, re-implement them through alternate pathways. Ideally, verify all experiments worth mentioning in the paper, or at least 75% of them.

### Novelty

A common and confusing requirement for papers is that the results be novel, something that is not covered before. What exactly does this mean? Science is fundamentally about **building a large body of knowledge**. This means that your work exists in the context of what has come before. **Novelty means it expands our knowledge**.

The conventional definition of novelty can be annoying and, in my opinion, focuses too much on shininess and doesn't capture the more important aspect of whether our knowledge has expanded. Another way to put this is: Should I assign different probabilities to propositions I care about after observing the results of this paper?

Rigorous, at-scale replications of shaky results, negative results of seemingly promising hypotheses, and high-quality failed replications of popular papers are all very valuable contributions. I would personally consider these novel because they expand our knowledge. However, the revealed preferences of many reviewers and researchers suggest they do not feel the same way. Such is life.

I don’t want to go too far re criticising novelty: there are many cases where I am uninterested in a paper due to lack of novelty. This primarily occurs with methods that I expect to work when applied in standard settings, and I assign a high probability of success, so the project provides few bits of information. While *knowing*that such a method failed could be interesting, projects can also fail due to researcher incompetence or bad luck. Therefore, it is difficult to draw meaningful conclusions without evidence of researcher competence.

Leaving that aside, novelty can be hard to communicate. Given a paper on its own, it's difficult to tell what is and is not supposed to be novel:

- Are the techniques used innovative or just standard techniques?
- Does the claim represent a deep conceptual breakthrough?
- Is it a very simple extension of standard ideas?
- Is it a natural consequence of a more ambitious claim put forward in a different piece of work?

The main way to address this is to be extremely clear about what is and is not novel, especially in the introduction and related work, and to liberally cite the most relevant papers and explain why your work is and is not different.

How to find out what came before? **Use a large language model**. If you’re not already familiar with a relevant literature, LLMs are pretty great at doing quick literature reviews, e.g. [Gemini Deep Research](https://gemini.google/overview/deep-research/?hl=en-GB)[[4]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fn29hhedam5nx). Reading the literature yourself is much better of course, but takes way, way longer and should have been done at the start project.

One reason this is very important is that, depending on what’s claimed as novel, the same paper could be perceived as either inappropriately arrogant or making a modest incremental contribution, depending on how the claims are presented.

Contextualizing your work within existing literature is **particularly crucial for experienced researchers** who are familiar with the field. Clear explanation in the introduction helps them quickly engage with your work and see what’s interesting, else it blurs into all other superficially similar papers they’ve read and doesn’t seem worth the effort.

There are a few problems with novelty as it is traditionally thought of

- **Novelty is often overemphasized** - it incentivises going for ambitious but shaky claims over simple and rigorous insights.
	- This can mean that if there's an existing paper that provides a preliminary but shaky case for a claim, going and doing it properly can seem less exciting, even though this is in some ways a more useful scientific contribution, as it establishes a confident foundation for others to build upon.
- Another complex question arises when you have legitimate complaints about prior work, and your work superficially looks derivative, but this is because you identified a significant methodological flaw or bug.
	- I recommend being clear that you have criticism, but it's important to remain professional while explaining what was flawed and why this matters, and how your work resolves it, without critiquing the authors or their motivations.
- There are various social norms that are kind of annoying, such as citing being obliged to cite the first instance of a concept (even if later iterations are much clearer) and referencing a ton of vaguely relevant work even if it adds nothing to the paper - people can get offended if not cited. But this doesn’t detract from all the ways that citations genuinely strengthen a paper

I personally prefer to just do work that is optimised for scientific value, and shoe-horn it into a peer review friendly lens at the end, if applicable. But I’m in a fortunate position here, and there are real career incentives around getting published.

Two example papers of mine where being clear about novelty was tricky:

- In [my Othello work](https://arxiv.org/abs/2309.00941), I built directly on Kenneth Lee's paper that showed an Othello plane model had a world model found with non-linear probes. My contribution was demonstrating that it could be found with linear probes, which was interesting for a bunch of reasons to do with the linear representation hypothesis, but I needed to be careful to *not*claim credit for anything Kenneth did
- In [my refusal paper](https://arxiv.org/abs/2406.11717?), our key result was that refusal is mediated by a single direction. But it was *not*novel to find that *a*concept was linearly represented, the significant part was doing it for refusal: a particularly interesting concept.
	- Other work had loosely tried to do this for refusal, but had less compelling results, so we had to explain why our’s was better (much larger effect sizes, more models, downstream tasks, etc)
	- We also showed that we could now jailbreak the model by removing this direction from the weights - the novelty was less that we could jailbreak models (that's already known to be easy with finetuning), but that we could do it with interpretability tools, and so cheaply, one of the first practical applications of interpretability (even if, you know, not quite for safety...)

### Rigorous Supporting Evidence

**A paper is worth little unless it can*****convince***[[5]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnc844hgv7nzl)**the reader of its key claims**. To do this, you need evidence. In machine learning, this typically means experiments. Below, I discuss how I think about what good experimental evidence looks like - see [my research process sequence](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand) for more advice.

With claims the priority is being able to communicate the intuitions to everyone, but with experiments **the priority is being able to justify it in full technical detail to an engaged, skeptical reader**. You also want to explain what’s going on intuitively, to support your claims, but this is less key than actually having good, legitimate evidence.

A particularly important thing to get right is **extensive red-teaming**: you should spend a good amount of your time, both during the original research and now, red teaming your narrative. One of the main traps introduced by the framing of “find a great narrative” is the temptation to ignore inconvenient contradictory results - don’t let this happen to you. Tips:

- Assume you've made a mistake - what is that mistake? Assume there's a hole in your case that your evidence supports your grand narrative - where is that hole? Try to break it apart.
- Try to get other researchers, especially more experienced ones, to weigh in.
- Make sure to extensively discuss limitations. If you notice issues, design and perform new experiments to test for them. This is all the more important the more ambitious or surprising your claims are.
	- When I read a paper with a bold claim, I have a strong prior that it is false, and I am constantly looking for holes. If I identify one, and the authors have not checked whether that is a real flaw, I will generally move on.
	- However, if they have preempted me and provide sufficient evidence that I can be confident it's not a flaw, then those papers can be incredibly exciting and insightful.

What does good evidence look like?

- **Good experiments distinguish between hypotheses**: Often, you will have several plausible hypotheses for some phenomena. The point of an experiment is to have results that vary significantly depending on which is true (i.e. that provide Bayesian evidence) - if the results vary enough, and the experiment is sufficiently reliable, then one good experiment can falsify many hypotheses
- **Can you trust your results?:**
	- **How reliable is my experiment?** Ask yourself: "How surprised would I be if it turned out to be complete bullshit due to a bug, error, noise, misunderstanding, etc.?" Investigate the most uncertain bits
	- **How noisy is my experiment?** If you ran similar experiments several times, how confident would you be that the results would be consistent? What is your sample size? What is your standard deviation? Are your results clearly distinguishable from noise? (There's a whole host of statistical theory here; your favourite LLM can probably effectively teach you about the basics.)
	- **Statistical rigour**: If you're doing some type of frequentist test[[6]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnn6o4myp4h8r), you probably shouldn't use p < .05 as a threshold. In stats heavy fields, like the social sciences, papers that report their central finding at .01 < p < .05, usually fail to replicate. If you're doing an exploratory approach, you should be skeptical of any result that isn't p < .001, as the number of possible hypotheses is vast.

		- Prior work discussing replicability is very strict on this point: "One prior study of 103 replication attempts [in psychology] indeed found a 74% replication rate for findings reported at p ≤ .005 and a 28% replication rate for findings at .005 < p < .05 (Gordon et al., 2021)". There are also various statistical reasons why true findings usually won't produce .01 < p < .05.[[7]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fni21bv7c6wsm)
	- This skepticism and sanity checking is especially key for particularly surprising or novel bits of evidence. Wherever possible, I will try to re-implement a key experiment from scratch or try to get at the same evidence via a somewhat different route, just to make sure that I'm not missing something crucial.
- **Ablation studies**: When a paper introduces a complex new method, there are often several moving parts. For example, they may make changes A, B, and C to standard practice. If they then only evaluate the standard method or the method with all three changes, it's impossible to tell which changes are actually effective and necessary. It's good practice to remove one change at a time, observe its effect, and then repeat this process for each change.
- **Unknown Unknowns:**How confident are you that there isn't some alternative explanation for your results that you're missing?
	- This is a gnarly one. You'll want to think hard about it, ideally ask other people for feedback and get their perspectives. However, ultimately, you may sometimes just need to move on after a reasonable effort and accept that you may have missed something.
- **Avoiding Misleading Evidence (Cherry-Picking and Post-Hoc Analysis):**
	- **Was this cherry-picked?** Researchers can, accidentally or purposefully, produce evidence that looks more compelling than it actually is. One classic way is cherry-picking: presenting only the examples that look most compelling. This is particularly dangerous with qualitative evidence, like case studies.
		- While qualitative evidence can be extremely valuable, it’s important to note *how* cherry-picked it was. Ideally, provide randomly selected examples for context to give a fairer picture.
		- The main exception is if your claim is an existence proof. In this case, one example suffices, if it’s a trustworthy result.
	- **Track pre/post-hoc analysis.** It's important to clearly track which experimental results were obtained *before* versus *after* you formulated your claim. Post-hoc analysis (interpreting results after they're seen) is inherently less impressive than predictions confirmed by pre-specified experiments.
		- Be aware that even complex predictions suggested by a hypothesis can turn out to be correct for the wrong reasons
		- For example, in [a toy model of universality](https://arxiv.org/abs/2302.03025), I came up with the key representation theory-based algorithm I thought the network would follow before we got our key pieces of empirical evidence. I felt very confident. However, follow-up work found that a different explanation, which also involved representation theory, was what was actually occurring.
- **Quality Over Quantity:**Try to prioritise having at least one really compelling and hard to deny experiment, over a bunch of mediocre ones.
	- If you do have many experiments, often some are more compelling than others. Highlight the ones that most strongly support your claims in the main text and consider moving others to an appendix or referencing them more briefly.
- **Diverse Lines of Evidence Are Robust:**On the flip side, it can be far better to have several *qualitatively different* lines of evidence all pointing to the same conclusion, rather than many very similar experiments that all use similar methodologies and standards of proof.
	- Qualitatively different basically means “given the result of experiment 1, how well can I predict the result of experiment 2?”
	- This can justify putting effort into weak lines of evidence; for example, qualitative analysis of some data points can be useful supporting evidence of a quantitative study, even if insufficient to carry a paper independently, as they make it less likely that the summary statistics hid a subtle flaw.
- **Distilling Experiments:**
	- Often, at the end of a project, you'll have run many experiments, some of which felt around the edges of your core claims. But by this stage, you likely have a much clearer idea of what the most promising kinds of evidence are. If practical (considering time and resources), consider going back to run a more conclusive, decisive experiment using what you now know.
	- This could also involve scaling up: using more models, larger sample sizes, sweeping hyperparameters more thoroughly, running on more diverse datasets, etc.
- **Baselines are Crucial:**
	- A common mistake is for people to try to show a technique works by demonstrating it gets "decent" results, rather than showing it achieves *better* results than plausible alternatives that people might have used or are standard in the field.
		- Implicitly you’re supporting the weak claim “method X works at all” not “method X is actually worth using in practice”
	- Sadly this is especially prevalent in fields like mechanistic interpretability, where the comparative need for qualitative evidence can lead to neglecting more rigorous and systematic quantitative comparisons against strong baselines - the best papers have both qualitative and quantitative evidence.
	- **The subtlety of baselines:** It's not enough to just *have* them; you must strive to have the *strongest* possible baselines. Put meaningful effort into making them good. Often, a "competitor" method can seem weak but can significantly improve with proper hyperparameter tuning, prompt engineering, or appropriate scaffolding.
		- There's a natural bias to invest more effort in making one's "cool, shiny" new technique look good than in optimizing "boring" baselines. Resist this. Rigorous comparison to strong baselines is critical for good science and for genuinely persuading informed readers.
- **The Guiding Question for Evidence:**
	- Ultimately, the question to ask about your evidence is: "Should this update a reader's beliefs about my claims?" not "Does this fit the stereotypical picture of a rigorous academic paper?" While the latter often correlates with the former, your primary goal is genuine persuasion through sound evidence.
	- Eg if reading several dataset examples by hand is genuinely strong evidence of your claim, just report that and justify why it’s great evidence!
- **Reproducibility & Publishing code**: Rigour can be in the eye of the beholder: if readers cannot understand or verify it for themselves, it’s far harder to consider it rigorous. So your paper will be made substantially more useful by providing more detail about your exact methods.
	- A particularly useful approach is sharing your code. This enables others to build on your work and clarifies any ambiguities left in the paper (there will always be some). More broadly, it provides transparency into your exact process.
	- If you have time, you should:
		- Ensure the codebase runs on a fresh machine
		- Write a helpful README that includes links to key resources like model weights or datasets (which can be easily hosted on Hugging Face)
		- Create a Python notebook demonstrating how to run the key components

Tragically, the world is complicated, and there is often no single clear recipe to deal with all edge cases in research. These considerations are guidelines to help navigate that complexity

### Paper Structure Summary

How are these claims and experiments translated into a paper?

- **Abstract**: Motivate the paper, present your narrative and the impact: explain your key claims and why you believe they are true - be as concise and high-level as possible, while still getting the point across. Give enough information for a reader to understand the key takeaways of your paper and whether to read further or not - they often won’t read further, so it’s key that they still get the gist!
	- The reader is coming in from a cold-start, and may have no idea what your paper is about - you need to help them orient *fast*, and indicate what “genre” your paper fits into
	- The rest of your paper exists to support the abstract
- **Introduction**: Basically an extended abstract that fleshes out your narrative - explain your key claims, motivate them, contextualise them in the *key*parts of the existing literature. Explain your key experiments and the results and why this supports your claims. Ensure the reader leaves understanding the narrative of the paper, and whether to read further or not - they often won’t. This is basically a self-contained paper summary, don’t worry about “spoiling” the paper or repetition - with a complex idea, you want to repeat it in varied ways so that it sticks.
	- The introduction sets the structure of the rest of the paper
- **Main content**: This is where the real technical detail lives. Clearly and precisely explain background concepts and results, your precise claims, what exactly you did for your experiments (in *full*detail, using appendices if need be, relevant baselines, etc), the results, what they mean and their implications, etc.
	- This should be tightly planned to support the key claims, not sprawling and comprehensive. **For section and subsection you should have a clear answer for how it contributes to the narrative**, and would be damaging to remove.
		- I recommend first planning out clear opening and closing sentences of each paragraph: what does the paragraph show and how does it fit into the paper?
- **Figures**: Figures and tables are a key medium for communicating experimental results. Diagrams are great for communicating key ideas and claims. Put a lot of effort into your figures.
	- Good captions are also crucial - you need to given context on what the figure shows, the nuance and intended interpretation, and key technical detail. Ideally the reader will understand everything from just the figure *and*just the caption, though this is ambitious
- **Related work**: This is a mini literature review - I generally put this after the main content and don’t think it’s super important. Giving context on a few key similar papers and how your work differs is crucial, but typically done in the intro.
- **Discussion** (/limitations/conclusion/etc): A place to put all the high-level reflections - limitations of your work, future work, key implications, etc. This is not essential, but is nice if you have something worthwhile to say. Acknowledging key limitations is very important, and papers that don’t do this are substantially weaker and less useful (in my opinion).
- **Appendices**: Everything else - in the main paper you need to care about being concise, but here you can do whatever you want. Often you want to briefly discuss something in the main paper and move all technical detail to the appendix. Appendices are pretty low stakes and rarely read except by superfans, so don’t stress them too much.

## Analysing My Grokking Work

This is all pretty abstract. To concretise this, let’s look at [my grokking paper](https://arxiv.org/abs/2301.05217) through this lens. I’ve broken it down into claims, evidence, context and motivation, with commentary thrown in. This is somewhat stylised for pedagogical reasons, but hopefully useful!

- **Meta**: This was a challenging paper to write!
	- There was significant technical detail to our claims and evidence, largely unfamiliar to readers - mech interp was very new, and we did something weird and novel. We needed to communicate our claims (the algorithm) and our experimental evidence, *and*justify why the evidence was believable, since there were no standard methods to follow
		- A good diagram was critical to explaining the algorithm:

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/65a5b7254debaeb354f0e6fd637b58c5abd07a533d472efb39d776799083c27f/tuw5q8dwrpyd3hegimjy)

*Figure 1 from the paper, and lead image in the tweet thread*

- This was more like two papers - the reverse engineering, and the study of circuit formation, and we needed to compress both into the same page limit. Fortunately, they did fit a cohesive theme

**Structure**:

- **Claim**: We fully reverse engineered a tiny transformer trained on modular addition
	- **Meta**: This is a general claim, but about a specific model
	- **Context**: We needed to explain the entire notion of reverse-engineering a model from its weights, as readers may not have been familiar. The motivation for why this is interesting is pretty obvious, but the goal is easy to misunderstand
	- **Evidence**: We show this with several lines of evidence: activation and weight analysis, and causal interventions
- **Claim**: This circuit forms gradually, well before the point of sudden grokking -> grokking is a gradual from memorisation followed by removing memorisation
	- **Context**: The entire notion of grokking as covered in prior work
		- **Meta**: This is critical context to understand my paper, but is also prior work - I need to explain enough detail for unfamiliar readers to follow, but without excessive repetition, and cite it for them to see further details.
	- **Motivation**: Grokking is a big deal and surprised many people. We show it’s fairly different from what people think.
	- **Evidence**: We show this by designing convincing progress measures to track the circuit, and show that they shift well before grokking

## The Writing Process: Compress then Iteratively Expand

***Note**: Check out*[*my paper writing checklist*](https://docs.google.com/document/d/1AoF6bPJp-muWnsZLMmfcxo1fmAu1izUzZXDFHar-35o/edit?tab=t.0)*for a concrete to-do list of what I recommend doing here*

So, you have a list of claims and key experiments. Now, all you need to do is write the paper! I recommend an iterative process - start with a bullet point narrative, then a full bullet point outline, then flesh it out into prose, taking time to reflect at each stage. See above or the next section for more details on the actual structure of a paper, here I just try to convey the high-level strategy

A key challenge in paper writing is the **illusion of transparency** - you have spent months steeped in the context of this research project. **You have tons of context, your reader does not**. You’ll need to help them understand exactly what you are doing, in the large space of all possible ML papers, and address all the misconceptions and possible misunderstandings, even though to you it all feels obvious. This is a difficult skill - wherever possible, get extensive feedback from others to address

**Spend far more time on early sections**: Realistically, tons of people read the title of your paper, many read the abstract, some read the introduction/skim the figures, and occasionally they read the whole thing. This means **you should spend about the same amount of time on each of: the abstract, the intro, the figures, and everything else**[[8]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fn8swe1i37vnm). (I’m only half joking)

### Compress

You should **start by compressing your work as much as possible**. Some tips:

- Verbally describe it to someone.
	- Bonus: Ask them what was most interesting, or to repeat it back to you
- Plan out a talk.
- Give your research notes to an LLM and ask it to summarize the key points.
- After each, think about what's missing, what's extraneous, what’s inaccurate or misleading, and iterate.

This compression step is crucial because it forces you to identify:

1. The 1-3 concrete claims you believe to be true
2. Why these claims matter (brief motivation)
3. The crucial experimental evidence for each claim (ideally 1-3 key experiments per claim)

Next, critically evaluate this compressed version:

- Do your experiments genuinely support your claims?
- Are there ways your experimental evidence could be flawed or misinterpreted?
- Could the evidence be true but the claim false? How?

As part of this process, write down common misconceptions, limitations, or ways someone might over-update on your work.

### Iteratively Expand

Once you have a compressed list of bullet points that you are satisfied with, you should start iteratively expanding and developing them. After each step, stop, reflect, read through, and edit - rushing a step can lead to a lot of wasted time at the next step.

If you have a research supervisor/mentor, it is very valuable to get feedback at each stage - I find it way faster *and*easier to give feedback on a narrative or bullet point outline than being sent 8 pages of dense prose! Even if you don’t have a mentor, try to get feedback from *someone*[[9]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnamgw22xkub).

1. Start with the **compressed bullet point narrative** - make sure you’re happy that this captures the narrative you want!
2. Write a **bullet point outline of the introduction** - **the north star here is to communicate what your claims are**, exactly, (including which parts are novel vs building on prior work), **why they matter**, and a high-level idea of **why they are true**
	1. This involves more detail, key citations to the literature, more detailed motivations, etc. Generally it won’t get too technical, but can involve explaining a few crucial concepts.
	2. Flow matters a lot here! Try to get feedback on how it feels to an unfamiliar reader, and how cohesive it feels
3. Write a **bullet point outline of the full paper** - covering the key experiments, results, methodology, background, limitations, etc
	1. **The north star is to convince a skeptical, engaged reader that your claims are true** - give them enough information to understand your experiments and the results
	2. **A good outline is tight and minimal** - every part of it should have a clear role in the overall narrative. If you don’t have a good answer to “what goes wrong if I cut this”, you should cut it.
	3. **Good figures are crucial**to communicate results - plan these out, but leave making them to step 4
	4. This can include writing the related work, or you can leave that to the end.
4. **Results**: Collect key experimental results and make first draft figures to show. Does this convincingly support your narrative? What’s missing? Which parts and complex and need more exposition, vs standard/unsurprising and can be sped through?
	1. You can’t always get this done in advance, but it’s *much*better if you do - more time to refine, iterate, etc.
5. **First draft**: Flesh this out into prose and full technical detail
	1. If you have writer’s block, try giving an LLM your outline, some relevant papers, and asking for a first draft. *Do not*just copy this into your paper, but I find that sometimes LLMs have good ideas, and that frustration with poor quality LLM write-ups can be a great way to break through writer’s block.
6. **Edit it**: Repeatedly pass over your first draft (and get feedback), clean it up, polish it, make the narrative as tight and clear as possible, cut out extraneous fluff, make the figures glorious, etc
	1. This is worth spending a *lot*of time on, it can make a big difference!

## The Anatomy of a Paper

OK, so what actually goes into a paper? What are the key components you’ll need to write, and what is the point of each?

### Abstract

*Check out the annotated abstract earlier for a concrete breakdown*

An abstract should **give a cold-start reader a sense of what the paper is about** - what sub-field, what type of paper, what key motivating questions, etc. This is a key manifestation of the illusion of transparency:  you know exactly what your project is about but to your reader there is a large space of possibilities, and without any context may have completely incorrect priors and wildly misinterpret.

People will often leave your abstract then move on, unless strongly compelled - it’s a big deal to get right, and deserves high polish

A common approach is:

- First sentence: Something uncontroversially true that clearly states which part of ML you're focused on (e.g., "Thinking models have recently become state-of-the-art across many reasoning tasks.")
- Second sentence: Something that makes clear there's a need, something unknown, or a problem for your paper to solve (e.g., "The transition to reasoning models raises novel challenges for interpretability.") - this should convey (some of) the motivation

Now the reader is situated, you need to *concisely* communicate your claims. Again, illusion of transparency - they often won’t know your techniques, the work you’re building on, key ideas, etc. **Abstracts should be as accessible as possible** - use simple language as much as you can

- Sentence 3: State the crucial contribution of this paper and why it is exciting - you’ll need to lose nuance, this is OK.
- Optional: Sentence 4 should provide clarifying details on that claim, such as its meaning and how evidence could be provided if not obvious.
	- Include key definitions for any necessary jargon, though jargon should be avoided if possible, unless it’s standard in the field and useful to contextualise the paper within the field.
- Each of the next few sentences should focus on either key experimental evidence or additional important claims. These can sometimes overlap, where a specific claim being true also supports the main claim.
	- Try to have 1 sentence per idea - this forces you to be concise, without getting overwhelming.
- If possible, include a concrete metric or result in any of the above that gives readers a sense that your results are real and substantial.
	- This can look like folding in key evidence of a claim into the sentence introducing the claim.

Finally, close with motivation:

- Final 1-2 sentences: Wrap up by reminding readers why the paper matters/is a big deal, its implications, and how it fits into the broader context.
	- This is also a good place to clearly state your standard of evidence, whether your work is:
		- A preliminary step towards…
		- Shows that method X should be used in practice
		- Shows that practitioners should take care when using method Y
		- Establishes best practices for Z
		- Provides compelling evidence that…

### Introduction

The introduction is broadly similar to the abstract but more extended and in-depth. I proceed in roughly this order:

- Paragraph 1: **Context** - What topic are we studying, what is the key motivating question, and why does it matter?
	- Optionally: 1 sentence on how our contribution answers it
	- It’s good to liberally cite papers here to establish things like ‘this is a real field’, ‘this problem matters’, ‘people are interested in it and have tried (and failed) to solve it/have solved variants’
- Paragraph 2: **Technical background** - what do we know about this problem? What are the established techniques our paper rests on? Etc
	- It’s good to cite liberally here to establish that what you’re using are standard methods and concepts, and to give the reader more context.
	- Here and in paragraph 1 you want to better situate your problem in the broader strategic picture of the field. Why does this matter? What other work has been done here, and why is it inadequate?
- Paragraph 3: **Key contribution** - What exactly is our main claim? Add key nuance, detail, context, etc.
- Paragraph 3.5[[10]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fn38hw5w3ydeg): **Our case** - summarise the most critical evidence we provide that our main claim is true
- [Optional]: More paragraphs for a second or third claim and the key case
- Paragraph 4: **Impact** - What should you take away from this paper? What are the implications, why is it a big deal, who should take different actions as a result of the results, etc. This may be emphasising practical utility, pushing forwards basic science, correcting common misconceptions, etc.
- **Contributions**: End with a bullet point list of concise descriptions of your key claims, ideally with concise descriptions of key evidence
	- You want something a reader can look at and decide if they’re impressed/interested

Note: Citing here isn't about performatively covering all the relevant papers[[11]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnqfem2sure9). It's about providing the context a reader needs to understand why your work is interesting and how it's limited. I try to have at least one citation for each step in an important argument, eg why

The introduction is where you have room to define key terms and concepts required to understand your claims, especially if they're somewhat technical.

It's often good to explicitly end with a bullet-point list of your contributions, which are basically just the concise claims you believe to be true, potentially with brief references to the supporting evidence.

### Figures

Figures are incredibly important. Having clear graphs can be the difference between a very clear and easy-to-read paper and an incomprehensible mess.

To create a good figure:

- Ask yourself, "What exactly is the information I would like someone to take away from this?" It's not just about finding the list of numbers output by your experiments and shoving them into some standard plotting software, you want to carefully choose a visualisation that emphasises the desired information and takeaway.
- Ask yourself, "Why does this experiment tie back to my core claims? How would I like the reader to interpret these results? Which parts do I want to draw their attention to?"
- Consider annotating a graph or, if there's one particularly important line, emphasising it
	- E.g. make it dark while the others are light and low opacity, or all the other ones of the same color.
- Include standard elements like axis titles, a clear caption that explains what the figure is, how to interpret it, or at least where in the text they should look to understand what's going on.
	- Make sure the axis title and ticks are large enough to read, and have a good clear legend.
- Often you can compress a fair amount of information into one graph - for example, using different sizes and shapes of markers on a scatterplot, different colors, etc.
- For heatmaps, if your data is positive and starts at zero, use a color scale that is white at zero and dark at the max (in plotly, "blues" is good). If your data is positive and negative with zero as a meaningful neutral point, use a color scale where zero is white (in plotly, "RdBu" is good).
- Avoid having reds and greens conveying key information, 4% of people are red-green colourblind

It can work well to combine several key graphs into one figure and make it your figure 1. E.g.:

[Language models represent space and time](https://arxiv.org/pdf/2310.02207):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/0c3d3f7b5a316bbc6ae17ed433c3de39c98ebbfc88b2b7ac629c8319c8c0954b/fqnwtulpjr6orkq1mp3j)

[Not all features are one-dimensionally linear](https://openreview.net/pdf?id=d63a4AM4hb):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/8e16a323eb589967f6f16d49476037cdb1bbd722cc1ed62c1fa420f5d73b4cbc/dcacm5v0fr2gtsc9dqdg)

Another kind of figure is an explanatory diagram rather than a graph. This can be a high-effort but very effective figure one, that gives people a sense of roughly what is happening in the paper. This should be something that would catch people's eye if you put it as the first image in a tweet thread about your paper. Some diagrams I liked (intentionally at several different levels of effortful):

[Emergent Misalignment](https://www.emergent-misalignment.com/):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/4d23b0480db579d24ff12c89031372b38f701222f393d197d6bd396a5bc03c87/hp4mfofcp4sdnn7loq4h)

[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/c76ea31c53e60cd88e4b4559ecdf4ccb53410d0c75cfe58c5990bcb7600ccb4b/s2xy3kemvalmdsw2y6gs)

[CoT in the wild is not always faithful:](https://arxiv.org/pdf/2503.08679)

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/6b9d49a36511ed41454f58686dad379917a4c473b5e7678bf66d98bd912d4e6e/xsqkni9iosbyt5mbdoet)

[Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717?):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/711cd46c3e32ab7ab39f28df69d0806d5451ce804556bba0db9a1870c690eedf/sjksooqcgwspk5cc43u8)

[My grokking modular addition work](https://arxiv.org/abs/2301.05217):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/cb657958d42a449c32dc6c7a4d4eec6e13407e8b585fd597799bf60dcbaaade5/edc4br2htkw6gxkgrsnz)

### **Main Body (Background, Methods and Results)**

Most of the actual paper, by word count, should be about **communicating your experiments and results in precise technical detail**. To do good science, it is important that researchers can understand exactly what you did and what you observed, so they can draw their own conclusions rather than needing to take things on faith. For example, in interpretability, there are ways that a method can give completely useless answers if misapplied, so it’s crucial that I know if a paper did that, even though the detail might seem totally unimportant to the authors!

Ideally, you want to communicate the information at several different layers of abstraction. It's your job to ensure that readers understand:

- The key background context required to disambiguate and understand your work - key terms, techniques, etc[[12]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fncamk72o22gu)[[13]](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fndj77nx985wb)
- What your results are and how to interpret those results and their significance
- What you actually did for your experiments
	- Why this was reasonable/well motivated/relevant to your claims
	- The specifics of various technical choices you made, and their implications for how to understand the results.

For structure, here’s a good default:

- **Background**: to explain the relevant context and terms - in particular, please define terminology and crucial techniques!
	- If pressed for space, you can put a glossary of key terms/definitions as an appendix, I always appreciate this
	- If you’re defining something new for this paper, put this in a section which is clearly *not*about reviewing known things (a new section or separate subsection)
- **Methods**: Explain the methods you used and why they are relevant to the problem
- **Results**: Specify exactly how the methods are applied as experiments, and what the results are
	- If you have a bunch of experiments using fairly similar methods, put each in a different subsection

If the experiments for each claim are more boutique, or if there are several claims with different styles of evidence, then I try to give each type of evidence its own section while explaining how it ties back to the overarching theme, rather than a methods -> results section. People will forget about the first method before they see its results.

### Discussion

Explaining the limitations of your work is a crucial part of scientific good practice. The goal of a paper is to contribute to our body of knowledge. Readers must understand the limitations of the evidence you provide to have a calibrated sense of what knowledge they have learned. And it's important that you put a good faith effort into documenting limitations because you know far more about your work than the readers, so they may miss things.

There is a common mistake of trying to make your work sound maximally exciting. Generally, the people whose opinions you most care about are competent researchers who can see through this kind of thing. And I generally have a much higher opinion of a piece of work if it clearly acknowledges its limitations up front. I’m not sure if this makes it easier or harder to get published.

This is also the place to discuss broader implications and general takeaways, future work you’d be excited about, reflections, etc.

Some people have conclusions too. Personally, I think conclusions are often kind of useless; the introduction should have explained this well. You can skip it

### Related Work

Generally, related work is often treated like a bit of an annoyance and afterthought. The feeling that you need to cite lots of things that aren't actually relevant can be annoying, but sometimes there is very important work that has done similar things to you, and a reader might have seen that and wonder why your paper is interesting.

It's very important to clearly explain why what you did is different or, if what you did is not very different, either acknowledge this ("that was parallel work") or explain why your work is still slightly interesting in this context, or how you fixed a mistake in prior work (stated politely).

But that said, there’s a lot of annoying norms here and related works often add little value IMO - needing to cite a lot so you look like you’ve put in enough effort, covering minor or obscure things that aren’t particularly relevant, citing low quality works to be polite, making sure to cite the first instance of each thing, etc. Contextualising in the literature is important, but ideally I’ve already covered it in the introduction.

Related work is often put as the second section of the paper. Personally, I generally prefer it to be the penultimate section. I think related work should only be upfront if it plays an important role in motivating the paper - if your paper is very heavily tied to the surrounding literature, plugging a gap, correcting a mistake, or unlocking a new capability that would enhance various bits of prior work.

### Appendices

Appendices are weird. They're basically the place you put everything that doesn't fit into the main paper. One way to think about it is that you're actually writing a much longer than nine-page paper - the main body *and* the appendices - but you've chosen a highlights reel for the first nine pages where you put all the absolutely key information. You place all the less crucial information in the appendices for readers to pick and choose from as they see fit.

In general, the crucial scarce resource you must manage is the reader's time and attention. The main body should be aggressively prioritized to make the most of this, be engaging, and communicate the most important pieces of information. But if you have a lot more to say than you can fit in there, then that's what appendices are for. A truly interested reader can go and take a look, though most won't.

Generally, appendices are held to a notably lower standard than the main body and will be read far less, so you should not feel obliged to put in meaningful effort polishing them. This is the standard solution to the dilemma when you want to include full technical detail but have done some fairly complex and convoluted work that just won't realistically fit.

## Common Pitfalls and How to Avoid Them

### Obsessing Over Publishability

Peer review is notoriously terrible for seeking truth. Reviewers often have biases, like favoring work that feels novel and shiny and exciting, or that doesn't feel weird or too new, or that doesn’t seriously challenge their existing beliefs. This has been shown in [rigorous RCTs](https://blog.neurips.cc/2021/12/08/the-neurips-2021-consistency-experiment/), where NeurIPS 2021 gave some papers two sets of reviewers and compared their decisions. The results… aren’t great:

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/eJGptPbbFPZGLpjsp/ovczzjp22rckjrhzecuo)

I personally think that, at least in safety, doing good work that people respect matters more than getting into conferences, though both are nice. I’ve generally had fairly good results with just trying to write high-integrity work that explains why I believe it is interesting, and just trying to do good science and the work that I think is highest impact, even if it doesn't fit the academic mold.

But it’s pretty plausible to me that many of the people reading this are not in such a fortunate position, and that getting first author papers into top conferences would be a meaningful career boost, especially your first 1-2 papers. The strategy I generally recommend for my mentees is to spend most of the project doing the best scientific work they can. Then, as we approach the end of the project, we figure out how to wrap it up in a maximally conference-friendly package while writing and submitting it. If we did anything that made the work noticeably worse, we can undo it before uploading to Arxiv.

### Unnecessary Complexity and Verbosity

Papers are seen as prestigious, formal, and highly intellectual artifacts. As a result, there's a tendency towards verbosity or trying to make things sound more complex and fancy than they actually are, so they *feel*impressive. I think this is a highly ineffective strategy. If I don’t understand a paper, I generally ignore it and move on, or assume it’s BS in the absence of strong evidence to the contrary. Often, the best papers just take some very simple techniques and apply them carefully and well. There’s a real elegance to being simple and effective.

People need to understand a paper in order to appreciate it and build on it and think it is interesting (except for superficial Twitter clickbait). Generally, you want to be precise, but within the constraint of being precise, be as simple and accessible as possible. Try to use plain language and minimize jargon except where the jargon is needed to precisely convey your meaning. You get points for quality technical insights, not for sounding fancy. Verbosity and overly complex language and jargon is actively detrimental to your paper’s prospects, IMO.

### Not Prioritizing the Writing Process

People often do not prioritize writing. They treat it like an annoying afterthought and do all the fun bits like running experiments, and leave it to the last minute. This is a mistake. Again, your work only matters if people read and understand it. Writing quality majorly affects clarity and engagement. Writing is absolutely crucial and is a major multiplier on the impact of your work.

I typically recommend that people switch from [understanding mode to distillation](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand) and paper writing a month before a conference deadline, if at all possible. You should want to spend a lot of your time iterating on a write-up, getting feedback, trying to make it clearer, thinking about weaknesses, etc.

## Tacit Knowledge and Beyond

One irritation I have about the standard paper structure is that it heavily incentivizes being rigorous and maximally objective and defensible. Obviously, there are significant advantages to this, but I think that often a lot of the most valuable insights from a research project come in the form of tacit knowledge.

This might be:

- This was hard and here are the steps we had to follow to get it to work.
- Here are some ways we noticed our experiments catching fire and what we did to fix them.
- Here's my fuzzy intuition of what's going on in the big picture - I can't fully defend it, but I'm reasonably confident this is true after several months of screwing around in this domain.
- Here’s something I misunderstood for months before it suddenly clicked
- Here’s a common misconception in this domain, or way people often misunderstand or overreact to our results
- Here’s my advice to anyone replicating this work, especially how to find hyper-parameters and deal with the fiddly bits
- Fleshing out a plan for future work directions you find particularly exciting.

I think this is really important, and I find it a real shame that this is often just discarded. I am personally a big fan of putting this kind of stuff as appendix A or as an accompanying blog post, where you can take as many liberties as you like.

## Conclusion

Your research will only matter if people read it, understand it, engage with it, and ideally believe it. This means that good paper writing is a crucial skill, but often neglected.

The core process should be to find the concise claims you believe to be true, the strongest experimental evidence that you believe builds a robust case for these claims, and use this to craft a coherent narrative. Then flesh this out into a bullet point outline of the overall post, reflect on it, and ideally get feedback, and iteratively expand.

Again, this is a highly opinionated post about how I personally think about the process and philosophy of paper writing. I'm sure many researchers will strongly disagree with me on many important points, and the correct approach will vary significantly by field and norms.

1. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefwwk8u16jdjf)** Note: I am in no way claiming that *I*follow this advice, especially in blog posts - this is my attempt to paint a platonic ideal, and advise on how to be a better person than I. Personally, I find actually writing academic papers pretty frustrating and much prefer blog posts
2. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnref5kbv6xekx6j)** I think writing great papers certainly helps, and if you’re new I recommend just trying to write the best paper you can, but there’s still a lot of depressingly perverse incentives from ML peer review
3. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefxjcznpnz657)** I mention this purely for completeness. I have never seen a convincing guarantee in deep learning, neural networks are far too squishy
4. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnref29hhedam5nx)** OpenAI’s is also good, but Google’s is free!
5. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefc844hgv7nzl)** Well, at least “provide enough evidence to somewhat update their beliefs”, convince is a fairly high bar
6. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefn6o4myp4h8r)** Which, admittedly, is fairly rare in ML as far as I’m aware
7. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefi21bv7c6wsm)** Thanks to Paul Bogdan for these points on p-values
8. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnref8swe1i37vnm)** Also the title, though I haven’t figured out how to productively spend 20% of my time on that yet…
9. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefamgw22xkub)** Paper swaps are a great way to get feedback - find someone else also working on a paper and offer to give each other feedback. Even if you have less time before the paper deadline, this tends to be a mutually beneficial trade.
10. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnref38hw5w3ydeg)** This can be its own paragraph, or part of paragraph 3
11. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefqfem2sure9)** Save that for the related work section…
12. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefcamk72o22gu)** If something is *super*widespread knowledge, no need to cover it, but err towards defining things. E.g. I wouldn’t bother defining the transformer architecture or an LLM, but I would define a sparse autoencoder or steering vector
13. **[^](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/eJGptPbbFPZGLpjsp#fnrefdj77nx985wb)** If this is too long, you can move most of it to an appendix

An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2
by Neel Nanda
7th Jul 2024
This post represents my personal hot takes, not the opinions of my team or employer. This is a massively updated version of a similar list I made two years ago

There’s a lot of mechanistic interpretability papers, and more come out all the time. This can be pretty intimidating if you’re new to the field! To try helping out, here's a reading list of my favourite mech interp papers: papers which I think are important to be aware of, often worth skimming, and something worth reading deeply (time permitting). I’ve annotated these with my key takeaways, what I like about each paper, which bits to deeply engage with vs skim, etc. I wrote a similar post 2 years ago, but a lot has changed since then, thus v2!

Note that this is not trying to be a comprehensive literature review - this is my answer to “if you have limited time and want to get up to speed on the field as fast as you can, what should you do”. I’m deliberately not following academic norms like necessarily citing the first paper introducing something, or all papers doing some work, and am massively biased towards recent work that is more relevant to the cutting edge. I also shamelessly recommend a bunch of my own work here, and probably haven't always clearly indicated which papers I was involved in, sorry!

How to read this post: I've bolded the most important papers to read, which I recommend prioritising. All of the papers are annotated with my interpretation and key takeaways, and tbh I think reading that may be comparable good to skimming the paper. And there's far too many papers to read all of them deeply unless you want to make that a significant priority. I recommend reading all my summaries, noting the papers and areas that excite you, and then trying to dive deeply into those.

Foundational Work
A Mathematical Framework for Transformer Circuits (Nelson Elhage et al, Anthropic) - absolute classic, foundational ideas for how to think about transformers. See my youtube tutorial (I hear this is best watched after reading the paper, and adds additional clarity). 



Deeply engage with:
All the ideas in the overview section, especially:
Understanding the residual stream and why it’s fundamental.
The notion of interpreting paths between interpretable bits (eg input tokens and output logits) where the path is a composition of matrices and how this is different from interpreting every intermediate activations
And understanding attention heads: what a QK and OV matrix is, how attention heads are independent and additive and how attention and OV are semi-independent.
Skip Trigrams & Skip Trigram bugs, esp understanding why these are a really easy thing to do with attention, and how the bugs are inherent to attention heads separating where to attend to (QK) and what to do once you attend somewhere (OV)
Induction heads, esp why this is K-Composition (and how that’s different from Q & V composition), how the circuit works mechanistically, and why this is too hard to do in a 1L model
Skim or skip:
Eigenvalues or tensor products. They have the worst effort per unit insight of the paper and aren’t very important.
Caveats (h/t Buck) - I think the paper somewhat overstates the degree to which we can fully and mathematically understand tiny attention-only transformers, and it may be worth checking out these critiques:
Its mathematical claim about one-layer transformers being equivalent to skip-trigrams is arguably wrong
Many people interpret the induction head hypothesis as being much stronger than evidence supports.
Understanding how a transformer works in detail is a pre-requisite for getting the most out of this paper, I recommend getting to the point where you can code a basic transformer (eg GPT-2) from scratch. I shamelessly recommend my Youtube tutorial on this (and accompanying tutorial notebook).
Superposition
Superposition is a core principle/problem in model internals. For any given activation (eg the output of MLP13), we believe that there’s a massive dictionary of concepts/features the model knows of. Each feature has a corresponding vector, and model activations are a sparse linear combination of these meaningful feature vectors. Further, there are more features in the dictionary than activation dimensions, and they are thus compressed in and interfere with each other, essentially causing cascading errors. This phenomena of compression is called superposition.

Toy models of superposition (Nelson Elhage et al, Anthropic) - absolutely foundational work on the idea of superposition, a ton of great ideas in there. I found reading it very useful for building my conceptual frameworks of what  My main critique is that they only study toy models (but, you know, it's in the name)

Deeply engage with:
The core intuitions: what is superposition, how does it respond to feature importance and sparsity, and how does it respond to correlated and uncorrelated features.
Read the strategic picture, and sections 1 and 2 closely.
Skim or skip:
No need to deeply understand the rest, it can mostly be skimmed. It’s very cool, especially the geometry and phase transition and learning dynamics part, but a bit of a nerd snipe and doesn’t obviously generalise to real models.
Finding Neurons In A Haystack (Wes Gurnee et al, during my MATS program). We try to find empirical evidence for and against superposition by identifying various attributes (eg “this text is in French” or “this is the second token in ‘social security’”) and then training sparse probes (ie at most k non-zero elements) on MLP neurons to find them. I recommend skimming over the sparse probing stuff and methodology, IMO the key stuff is the case studies, and building intuition for what's out there in models. Notably, we find monosemantic language neurons (eg is French), and evidence that compound word detection (eg “social security”) is done in superposition, distributed across many polysemantic neurons. Youtube walkthrough

I'm really proud of the exposition about superposition in appendix A, and think it’s highly worth reading, and clarifies some open thread left by Toy Models of Superposition
Fact Finding (Neel Nanda & Sen Rajamanoharan et al, Google DeepMind) - we tried to really understand how facts were stored in model MLP layers in superposition. Unfortunately, we failed - it seems cursed and complex. But I think we learned a fair bit in the process, provided further evidence that superposition is happening (it’s not just neuron aligned and there’s many more facts than neurons) and falsified some simpler hypotheses. The sequence is long, but I think it's worth reading post 1 carefully. Post 3 has the superposition relevant stuff

Note that this research was done before SAEs took off, and we do not use them.
Sparse Autoencoders
SAEs are a tool to interpret model activations in superposition - they’re a one hidden layer ReLU autoencoder (basically a transformer MLP layer), and are trained to reconstruct a model’s activations. L1 regularisation is applied to make the hidden layer activations sparse. Though not directly trained to be interpretable, the hope is that each unit (or feature) corresponds to an interpretable feature. The encoder + ReLU learns the sparse feature coefficients, and the decoder is a dictionary of feature vectors. Empirically, it seems to work, and I think they’re the one of the most promising tools in mech interp right now.

To understand the actual technique I recommend this ARENA tutorial, sections 1, 6 & 7 (exposition + code >> papers), but here are some related papers worth understanding. Note that all of these came out in the past year, this is very much where a lot of the mech interp frontier is at! However, our understanding of them is still highly limited, and there are many uncertainties and open problems remaining, and I expect our understanding and best practices to be substantially different in a year or two.


Towards Monosemanticity (Trenton Bricken et al, Anthropic): The foundational work in applying SAEs to models to find interpretable features, beautifully written, and with a lot of good exposition and tips for training SAEs well. I recommend reading in full. My main criticism is that they only looked at 1L models, so all the features lie in unembedding space (they get linearly mapped to the logits), so it’s not very surprising that they’re linear - but future work (below) shows their results hold up.

Sparse Feature Circuits (Sam Marks et al, David Bau’s group): A great initial attempt at circuit analysis with SAEs (ie using SAE features as circuit nodes). Uses attribution patching, introduces a variant with integrated gradients that empirically works better at early layers and seems fairly principled. I personally think the way they implement gradient computation is somewhat convoluted (lots of stop gradients) and it would be simpler to do it analytically (take the gradient with respect to the residual stream after the relevant layer, and then do linear algebra to find the gradient wrt an SAE feature)

Has some cool results applying their interpretability insights to reduce spurious correlations in a linear probe, reducing gender bias. I find this particularly exciting, as it feels like the start of a real-world application of SAEs, which I think is a big open problem for the field - if they are truly as useful as we think they are, they should be better than existing baselines in a fair fight on at least some real-world tasks. See this post from Sam on the long-term safety motivations here - we want to distinguish between behaviourally identical systems that tell us the truth vs telling us what we want to hear.
Transcoders Find Interpretable LLM Feature Circuits (Jacob Dunefsky, Philippe Chlenski et al, during my MATS program). Circuit analysis can either be causal intervention based (eg Interpretability in the Wild, or Sparse Feature Circuits), or weights based (eg Progress Measures for Grokking) - actually multiplying things out and seeing what happens. It’s very hard to do weights-based analysis on MLP layers in superposition though, as for any given feature, most neurons (and their non-linearities!) are implicated, and hard to decompose. Transcoders are an SAE variant that tries to solve this, by learning a sparse, interpretable replacement for an MLP layer, by training an “SAE” to map the MLP input to output. They seem to perform about as well as SAEs, and make circuit analysis much easier, though we still struggled to get true weights-based analysis working - many features have high alignment without co-occuring.

Interpreting Attention Layer Outputs with Sparse Autoencoders (Connor Kissane & Rob Krzyzanowski et al, during my MATS program): A nice post showing that SAEs can be applied to attention layer outputs, and that this just works (they do it pre W_O, which has nice benefits of being able to easily attribute things to heads). I particularly liked the focus on showing that these SAEs can be a useful tool for researchers, attacking problems previously out of reach. We found a deeper understanding of the semantics of the IOI circuit, got a sense of the kinds of features that form in attention layers, figured out the different roles of two induction heads in a layer, and roughly gauged the role(s) of every head in GPT-2 Small.

Towards principled evaluations of sparse autoencoders for interpretability and control (Alex Makelov & Georg Lange et al, during my MATS program). One of my favourite attempts to design better metrics for SAE quality, one of the major open problems in the field. We took the well-studied Indirect Object Identification circuit, trained SAEs on every head and layer, and evaluated two metrics: sparse control, whether the output could change from input A to input B by changing a sparse set of input A SAE features, and interpretability, whether we could use SAE features as probes to find the expected features. Because we roughly know what features to expect, we can also train dictionary vectors for them in a supervised way, measure the quality of these, and use this to get a baseline for our benchmark

One of the hard parts of this analysis is that, though we think we know what important features to expect in the IOI circuit (the value and position of the names), we can’t be sure we aren’t missing something (eg the gender of the names). We discuss this at length, and how we try to deal with it.
Gated SAEs (Sen Rajamanoharan et al, from my team at DeepMind): Introduced Gated SAEs, a new architecture for SAEs that gets similar reconstruction at half as many firing features while being either comparably or more interpretable. We also scaled SAEs to Gemma-7B. I (very biasedly) think this is worth reading as a good exemplar of how to rigorously evaluate whether an SAE change was an improvement, and because I recommend using Gated SAEs where possible.

Scaling and evaluating sparse autoencoders (Leo Gao et al, from the OpenAI superalignment team RIP 😢). Shows that top-k SAEs (ie replace the ReLU with “keep the top k pre-activations, set the rest to zero) are an effective technique, scale SAEs to GPT-4 (the largest model SAEs have been trained on!), do some rigorous exploration of SAE scaling laws, and propose several creative ideas for measuring SAE quality. I’m particularly excited about the ideas for how to measure SAE quality, since this is a big open problem in the field, and would be keen to see the metrics proposed fleshed out and applied in more detail. The work provided a feature viewer, but did little qualitative case studies itself - anecdotally the GPT-4 SAE features don’t seem that interpretable, but I haven’t explored this properly.

One big concern I had from this paper is how interpretable top-K SAEs are. Follow-up work from Anthropic showed that there’s no hit to interpretability, and confirmed that they are a significant performance improvement (comparable to gated SAEs)
Scaling monosemanticity (Adly Templeton et al, Anthropic) Similar to the OpenAI paper, the headline result is that SAEs scale to a (near) frontier model, Claude 3 Medium (Sonnet). But there’s a massive focus on case studies and qualitative analysis, which is very fun and worth reading, so it complements the OpenAI work nicely. They find a range of abstract and multimodal features eg an unsafe code feature that activates on pictures of “warning: this web page may be insecure”, and including some potential safety relevant ones. There’s a focus on showing that features are causally meaningfully variables and can be used to steer the model, similar to steering vectors.

The most famous part of this paper was Golden Gate Claude, the ultimate meme, an accompanying demo where they released a version of Sonnet that was steered with the Golden Gate Bridge feature to obsessively fixate on the bridge
A key nuance is that (in my opinion) the paper does not show that SAEs are the best way to steer (compared to lower tech methods like steering vectors, or even prompting). Rather, the goal was to show that SAEs do something real, by showing that they have an abstract, causal effect, and affect the model’s computation in sophisticated ways. I’ve seen many people see Golden Gate Claude and think they need an SAE to do steering, you probably don’t need to bother! 
There’s an accompanying blog post with a bunch of tweaks to significantly improve SAE training, which excitingly seem to stack with gated and top-K SAEs!
I really loved the feature completeness result - turns out that there’s a clean sigmoid relationship between the frequency with which a concept appears in the SAE training, the number of alive SAE features, and the probability that it’s learned by the SAE.
Activation Patching
Activation patching (aka causal mediation analysis aka interchange interventions aka causal tracing aka resample ablations - argh why can't we agree on names for things!) is a core mech interp technique, worth understanding in a lot of detail. The key idea is that, for a given model behaviour, only a sparse set of components (heads and neurons) are likely relevant. We want to localise these components with causal interventions. But, given any prompt, many model behaviours go into it. For example, if we want to know where the knowledge that Michael Jordan plays basketball lives, this is hard - we can do things like deleting components and seeing if the model still says basketball, but maybe we deleted the "this is about sports" part or the "I am speaking English part".

The key idea is to find contrast pairs - prompts which are as similar as possible, apart from the behaviour we care about, eg "Michael Jordan plays the sport of" and "Babe Ruth plays the sport of". If we patch activations from the Jordan token into the Ruth token (or vice versa), we control for things like "this is about sports" but change whether the sport is basketball or not, letting us surgically localise the right behaviour.

To understand the actual technique I recommend this ARENA tutorial (exposition + code >> papers), but here are some related papers worth understanding.


How to use and interpret activation patching (Stefan Heimersheim & me) An expository and highly opinionated note on how to think about activation patching. It’s a powerful but subtle and confusing technique, and we walk through some traps, how to use the technique well, and advice on interpreting results.
Note that this is not a paper, in the sense that it doesn’t produce original research, but rather an attempt to reduce Research Debt. I highlight it because I think it distils implicit ideas spread across many papers and the heads of many researchers in one place.
Causal scrubbing (Redwood) - This is my current favourite attempt to define a metric for how good your explanation of a model's behaviour is. The key idea is to identify all patches that are allowed by your explanation (eg resample ablating a head that isn't part of your hypothesis, or resample ablating the path between two heads that don't compose under your hypothesis) and to do all of them, and see how much damage it does. I found the sequence itself somewhat sprawling and hard to read, but found it valuable to get my head around the key ideas, and found it had multiple important methodological insights like the idea of resample ablation (rather than ablating a component by setting it to zero, replace its activation with the value taken from a random other input).
It's even stronger than just ablating all irrelevant nodes and edges - if some activation should have the same value on multiple prompts (eg "the previous token is ' cat'") then you are allowed to patch it between different prompts where the previous token is cat. And we can do this recursively. Unfortunately, this is a pain in the ass to implement, and I think you can basically get away with not doing it in practice?
Attribution Patching (me, work mostly done at Anthropic) - Activation patching at industrial scale, using a gradient based approximation. You can approximate the effect of patching a component from input A into input B with (act_A - act_B) . (grad_B). This works reasonably well in practice, and is blazingly fast - you go from doing a forward pass per component to doing two forward & one backward pass for every single component simultaneously! 
I think the key thing is just to know that the technique exists, but the post itself has a lot of good exposition on how to think about activation patching that I feel quite proud of. It doesn't do a good job of testing that attribution patching is actually a good approximation.
See the AtP* paper (Janos Kramar et al, Google DeepMind mech interp team) for a much more systematic set of experiments showing that attribution patching is a good approximation and the best way to use a limited compute budget (of several patching variants explored), and some improvements to the technique by handling saturated attention softmaxes better.

Marks et al (David Bau's group) introduces an integrated gradients based approach, which is slower, but seems to be superior.
Automated Circuit Discovery (Arthur Conmy et al) - The key idea is to take the kind of analysis done in IOI, finding the key sparse subgraph for a circuit, and automate it by doing it recursively and keeping the key nodes at each step. Importantly, it operates on edges, not nodes. Youtube walkthrough

I think the key insight is that this can be automated, and maybe to know that a specific method exists. I think the paper itself has lots of gory details that probably aren't worth engaging with in much detail?
Follow-up work (Aaquib Syed et al) showed that this can be done with attribution patching to be much faster
Distributed Alignment Search (Atticus Geiger et al). The key idea is that rather than activation patching a full component, which may not work if we aren't in a privileged basis, we can learn some subspace to patch instead! Honestly, I think this is the crucial idea: use gradient descent to find a subspace such that it has a causal effect when patched, and the rest of the paper isn't necessary. I also think that DAS can often be done with a 1D subspace, and the paper doesn't focus on getting the subspaces small. Atticus thinks about mech interp and causal graphs fairly differently from me, and engaging with how he frames things might be valuable for you! Youtube walkthrough (I've heard some people found the paper impenetrable, but found the walkthrough easy)
An Interpretability Illusion for Subspace Activation Patching (Aleksander Makelov & Georg Lange, during my MATS program) - A key accompaniment to thinking about distributed alignment search! It turns out that using gradient descent to find subspaces can be sketchy. Concretely, knowing that patching along a direction has a causal effect tells you that the direction both correlates with the feature in question and causally affects the output. if you have a direction that is dormant (matters causally but never changes) and a direction that is disconnected (causally irrelevant but correlates with the feature of interest), then their sum is both causal and correlated and so is valid to patch along. We show that you can find both of these directions everywhere in real models, and this illusion comes up in practice, especially when doing DAS on model layers rather than the residual stream. 

Narrow Circuits
A particularly important application of activation patching is finding narrow circuits - for some specific task, like answering multiple choice questions, which model components are crucial for performance on that task? Note that these components may do many other things too in other contexts, due to polysemanticity, but that is not relevant to this kind of analysis. At this point, there's a lot of narrow circuits work, but here's some of my favourites.

Note that this used to be a very popular area of mech interp work in 2023, but has fallen somewhat out of fashion. I am still excited to see work doing narrow circuits work in the SAE basis (eg Sparse Feature Circuits above), work using narrow circuits to understand or do something on real-world tasks, especially in larger models, and work automating circuit discovery, especially automating the process of finding the meaning of the circuit. But I think that there's not that much value in more manual Indirect Object Indentification style work in small models. 

Indirect Object Identification (Kevin Wang et al, Redwood) - The classic paper that used activation patching in detail to find a circuit used for a narrow task ("When John and Mary went to the store, John gave the bag to" -> " Mary"). I've gotten a ton of value out of thinking a lot about this circuit, replicating it, and thinking through the techniques. Just having a concrete example of a narrow circuit is high value. And I think there's several promising theories of change that factor through doing IOI style analysis on more important problems (eg deception, or 'why does Bing Chat gaslight users?')

I'm less confident it's worth a lot of effort close reading the actual paper lol, I think it was optimised for getting into ICLR, and there's a bunch of stuff on eg circuit completeness that isn't very interesting to me. 
Notable for being the first place I saw negative heads and backup heads studied, though IMO those are now best studied by reading copy suppression and the hydra effect respectively
The actual method used in the paper is mean ablation (mean taken over another, similar distribution) of circuit edges that they don't think are important. It's not clear to me that I'd use this myself (compared to eg patching between prompt pairs, possibly patching nodes rather than edges). I'd favour going through this tutorial to think about how to do IOI.
A Greater-Than Circuit (Michael Hanna et al, Redwood) - There's been a lot of IOI follow-up works, that look at circuits on other narrow tasks. This is one of my favourites, which includes some analysis of specific neurons. 
Task: "The war was fought from the year 1745 to 17" should be followed by 46 onwards not 44 or before
Does Circuit Analysis Interpretability Scale? (Tom Lieberum et al, DeepMind). Shows that IOI-style analysis scales by doing it on Chinchilla (70B) on the syntactic part of doing multiple choice questions - converting knowledge of the factual answer to the correct letter (not on finding the factual answer in the first place). Things basically scale! It's slow and painful and there's weird cursed shit (like heads that move the answer label to the end if and only if it is both correct and C), but nothing fundamentally breaks. I also think it's a fairly well presented and clean example of circuit analysis. 

I'm proud of the circuit analysis in my Fact Finding sequence (the focus of post 2), I think it's particularly clean and well-presented, and turned out fairly nicely.
Paper Back-and-Forths
One of my favourite phenomenons is when someone puts out an exciting paper, that gets a lot of attention yet has some subtle flaws, and follow-up work identifies and clarifies these. Interpretability is dark and full of terrors, and it is very easy to have a beautiful, elegant hypothesis that is completely/partially wrong, yet easily to believe by overinterpreting your evidence. Red-teaming your own work and being on guard for this is a crucial skill as a researcher, and reading examples of this in the literature is valuable training.

ROME + Follow-ups: A valuable thread in interpretability illusions. Rank-One Model Editing was a proposed factual editing technique that could insert facts into model MLP layers, by optimising for a rank one edit that would make the model say "The Eiffel Tower is in the city of" -> " Rome" (the optimisation target was that it would output Rome). The paper got a lot of attention, and the edit had fascinating effects, eg changing other answers about the Eiffel Tower to be consistent with the knowledge that it was in Rome. Can you see the flaw with this scheme?

My answer in rot13: Gur pber ceboyrz vf gung gurl qvq snpg vafregvba, abg snpg rqvgvat. Gb rqvg fbzrguvat lbh zhfg qryrgr gur byq guvat naq vafreg arj vasbezngvba, ohg EBZR unf yvggyr vapragvir gb rqvg jura vg pbhyq whfg vafreg n ybhq arj snpg gb qebja bhg gur byq bar. Guvf zrnaf gung gur ynlre vg jnf vafregrq va qvqa'g ernyyl znggre, orpnhfr vg whfg unq gb qrgrpg gur cerfrapr bs gur Rvssry Gbjre naq bhgchg Ebzr. Shegure gurer jrer yvxryl cngubybtvpny rssrpgf yvxr bhgchggvat 'ybbx ng zr' fb urnqf jbhyq nggraq zber fgebatyl gb gur Gbjre gbxra naq guhf bhgchg Ebzr zber, guvf yrq gb pregnva ohtf yvxr "V ybir gur Rvssry Gbjre! Onenpx Bonzn jnf obea va" -> " Ebzr". 
ROME (Kevin Meng & David Bau et al) - Given the above, I don't think it's worth engaging deeply with their method/tests, except as a pedagogical exercise. They also used causal tracing/activation patching to great effect, which legitimately was very cool, and significantly influenced the field.
Does Localization Inform Editing (Peter Hase et al) A follow-up showing that there was no correlation between the layers that were easiest to edit and the layers that mattered when patched. Unsurprising given the above! (I think the key info is given by the summary, and wouldn’t prioritise reading it deeply)
Detecting Edit Failures in Large Language Models (Jason Hoelscher-Obermaier & Julia Persson et al) A follow-up showing the "The Louvre is cool. Obama was born in" -> " Rome" example (unsurprising given the above!). (I think the key info is given by the summary, and wouldn’t prioritise reading it deeply)
The Interpretability Illusion for Subspace Patching paper discussed above shows (and almost proves) that there will exist such directions in any MLP layer before the fact retrieving heads
Othello + my follow-up: There was a very cool paper from Kenneth Li that trained a model to predict the next move in the board game Othello (like chess/Go) on synthetic games that randomly chose legal moves. They found that the model spontaneously learned a model of the board state internally, this could be probed for, and could be causally intervened on (with a very complex, galaxy brained method) and change model outputs in the desired way. Crucially, they found that non-linear probes (one hidden layer MLPs) could find the board state, but linear probes could not!

In my follow-up, I found that there was a linear world model hiding beneath! But that rather than saying whether a square was black or white, it said whether it had the current or opposing player's colour. Once you have this linear world model, you can causally intervene with simple vector arithmetic!
I think this is most exciting as real evidence for the linear representation hypothesis - the original paper seemed like a serious shot at falsifying it, and my follow-up showed that it survived falsification!
This paper brings me joy for (along with many other papers) conclusively falsifying the strong stochastic parrots narrative.
Emergent World Representations (Kenneth Li et al) Given the above, I don't actually think it's worth reading closely, except as a pedagogical exercise, but it's a cool paper!
Actually, Othello-GPT Has A Linear Emergent World Representation - the blog post form of my follow-up. Not obviously worth reading given the above, but I really like them, and think it has cool flow and research exposition. Post 3 has a blow by blow account of my research process and key decisions made at each point. 

Thanks to Andrew Lee, there's also a paper version - it's more rigorous, but less chatty and maybe less pegagogically good.
Bonus
I don't think the papers in here are essential reading, but are worth being aware of, and some are broadly worth reading if you have the time, especially if any specific ones catch your eye!

The induction heads paper (Catherine Olsson et al, Anthropic) - nothing important in mech interp has properly built on this IMO (the idea of induction heads is very important to know, but was introduced in A Mathematical Framework shortly beforehand), but there's a lot of cool stuff - a circuit was universal across scales, a circuit was crucial to in-context learning, phase transitions, bumps in the loss curve, etc.


Deeply engage with:
Key concepts + argument 1.
Argument 4: induction heads also do translation + few shot learning.
Getting a rough intuition for all the methods used in the Model Analysis Table, as a good overview of interesting interpretability techniques.
Skim or skip:
All the rigour - basically everything I didn’t mention. The paper goes way overboard on rigour and it’s not worth understanding every last detail
The main value to get when skimming is an overview of different techniques, esp general techniques for interpreting during training.
A particularly striking result is that induction heads form at ~the same time in all models - I think this is very cool, but somewhat overblown - from some preliminary experiments, I think it’s pretty sensitive to learning rate and positional encoding (though the fact that it doesn’t depend on scale is fascinating!)
Progress Measures for Grokking via Mechanistic Interpretability (Neel Nanda et al) - nothing important in mech interp has properly built on this IMO, but there's just a ton of gorgeous results in there. I think it's the one of the most truly rigorous reverse-engineering works out there, and the connections to phase transitions and explaining grokking were really fucking cool. Youtube walkthrough. See my blog post for more takes.

Also a good example of how actually understanding a model can be really useful, and push forwards science of deep learning by explaining confusing phenomena like grokking.
See this comment by Jason Gross suggesting some other highly rigorous works of mech interp on algorithmic models
Deeply engage with:
The key claims and takeaways sections
Overview of the modular addition algorithm
The key vibe here is “holy shit, that’s a weird/unexpected algorithm”, but also, on reflection, a pretty natural thing to learn if you’re built on linear algebra - this is a core mindset for interpreting networks!
Skim:
Reverse engineering modular addition - understanding the different types of evidence and how they fit together
Evolution of modular addition circuits during training - the flavour of what the circuits developing looks like during training, and the fact that once we understand things, we can just literally watch them develop!
The interactive graphics in the colab are way better than static images
The Phase Changes section - probably the most interesting bits are the explanation of grokking, and the two speculative hypotheses.
Logit Lens (nostalgebraist)

A solid early bit of work on LLM interpretability. The key insight is that we interpret the residual stream of the transformer by multiplying by the unembedding and mapping to logits, and that we can do this to the residual stream before the final layer and see the model converging on the right answer
Key takeaway: Model layers iteratively update the residual stream, and the residual stream is the central object of a transformer
Deeply Engage with:
The key insight of applying the unembedding early, and grokking why this is a reasonable thing to do.
Skim or skip:
Skim the figures about progress towards the answer through the model, focus on just getting a vibe for what this progress looks like.
Skip everything else.
The deeper insight of this technique (not really covered in the work) is that we can do this on any vector in the residual stream to interpret it in terms of the direct effect on the logits - including the output of an attn or MLP layer and even a head or neuron. And we can also do this on weights writing to the residual stream, eg the output weights of a neuron or SAE feature. This is called direct logit attribution, and is a really powerful technique.
Note that this tends only to work for things close to the final layer, and will totally miss any indirect effect on the outputs (eg via composing with future layers, or suppressing incorrect answers)
Steering vectors: A family of papers exploring the idea that language models can be controlled by adding vectors to their activations, and that these vectors can often be very cheaply found, eg taking the mean difference between activations with and without some property. I consider this more model internals work than mech interp, but I think it's worth being aware of, 
Activation Addition (Alex Turner et al, done with his MATS scholars before joining Google DeepMind) - The most conceptually simple version of this. You can eg take the difference in residual streams for "I love you" and "I hate you", add this vector in on a benign prompt like "I went up to my friend" and the model gets incredibly angry.
Inference-Time Interventions (Kenneth Li et al, Harvard) - Parallel work to Activation Addition, which found a truthful vector that improved TruthfulQA performance and reduced hallucinations

Representation Engineering (Andy Zou et al, a CAIS project) Following the above, this was a nice distillation of the ideas, that applied it to a range of more realistic settings

Refusal is Mediated by A Single Direction (Andy Arditi et al, during my MATS program). By applying existing techniques, we show in detail that chat-tuned models decide whether to refuse a harmful request (eg "How do I make a bomb?") via a single refusal vector - adding this to harmless prompts means they're refused, and ablating this on harmful prompts means they aren't refused. You can ablate this from the weights of the model to jailbreak it - it now rarely refuses things, has minimal damage done to its performance. This jailbreak is competitive with finetuning while being a fair bit easier, and I think is notably for being one of the more compelling real-world uses of model internals work so far.

The Hydra Effect (Tom McGrath et al, Google DeepMind) - makes the crucial point that self-repair aka backup in language models is a real thing (ie you delete a component and later components shift behaviour to compensate for its loss), but the paper is far longer than needed to get the key point and you can skim/skip most of it. Also observed in Interpretability in the Wild. Note that this happens even in models trained without dropout

Explorations of Self-Repair in Language Models (Cody Rushing et al, during my MATS program) A follow-up paper showing that self-repair happens across the full data distribution (not just narrow datasets), and exploring some of the mechanisms. Generally, self-repair is a mess, due to a range of mechanisms, only some of which we explore/catalogue. Notably, self-repair can be partially explained by the final LayerNorm's scale (if several components agree, and you delete one in a way that reduces the residual stream norm, the others get comparatively scaled up) - this explains a fraction of self-repair, up to 30% in extreme cases.
Copy Suppression (Callum McDougall, Arthur Conmy & Cody Rushing et al, during my MATS program). Really cool paper that IMO comes closest to really understanding a model component on the full pre-training distribution: we show that the main role of head L10H7 in GPT-2 Small is copy suppression: notice if earlier layers have decided to predict a token, check if that occurs earlier in the context, and if so attend to it and suppress it. Youtube walkthrough

The analysis is less complete than we hoped. We couldn't find any examples of it doing anything else, which I take as a big deal, but we only preserve 77% ish of the effect of the head when we ablate all other behaviours, and it's substantially worse if we restrict the query to just be the unembed of the relevant token. I don't know how to interpret this. 
Copy suppression heads have occured in the literature before as anti-induction heads (doing induction but suppressing the answer) and negative name movers in IOI - both were instances of this general algorithm!
Copy suppression explains part of self-repair - if you're suppressing an earlier prediction, and it goes away, then there's nothing to suppress.
Copy suppression is an important part of overall model calibration, and loss gets worse without it.
Linear Representations of Sentiment (Curt Tigges & Oskar Hollinsworth et al, during my MATS program) Really fun paper: we found that there's a "first principal component of sentiment" that seems to be shared across a wide range of sentiment-y tasks, matters across the pre-training distribution, matters causally on Stanford Sentiment Treebank, seems universal across models (we checked up to 7B), and can be found with a range of techniques which all broadly agree. To some degree, the paper is a model of doing a deep dive into interpreting a particular, interesting direction.

We also found the "summarization motif", where models store information about a sentence or phrase on punctuation like commas or full stops, and this matters causally for downstream effects.
Softmax Linear Units (Nelson Elhage et al, Anthropic) - We introduced a new activation function which we originally thought made MLP neurons monosemantic, but turned out to help a bit, but also to have a lot of superposition hidden under the hood. Not worth reading in detail, but worth it for grokking that illusions are everywhere in mech interp: we thought we'd solved superposition, but actually it was just smuggled in. Also for the section called qualitative results with a ton of cool examples of features

Language models can explain neurons in language models (Steven Bills et al, OpenAI) - Using GPT-4 to explain GPT-2 neurons. I wouldn’t prioritise reading in detail, but the core idea (and that GPT-4 is kinda good enough!) is very cool, and it's an important tool to be aware of. The work was a bit ahead of its time, as it came out a few months before SAEs became a big thing. Neurons are often not very interpretable, so the technique didn't work great, but SAE features are more interpretable so the technique is far more useful there.

An Interpretability Illusion for BERT (Tolga Bolukbasi et al, Google)
Good early paper on the limitations of max activating dataset examples - they took a seemingly interpretable residual stream channel (or neuron, note that it’s not an MLP hidden layer neuron) in BERT and took the max activating dataset examples on different (small) datasets, and observed consistent patterns within a dataset, but very different examples between datasets
Within the lens of the Toy Model paper, this makes sense! Features correspond to directions in the residual stream that probably aren’t neuron aligned. Max activating dataset examples will pick up on the features most aligned with that neuron. Different datasets have different feature distributions and will give different “most aligned feature”
Further, models want to minimise interference and thus will superpose anti-correlated features, so they should
Deeply engage with:
The concrete result that the same neuron can have very different max activating dataset examples
The meta-level result that a naively compelling interpretability technique can be super misleading on closer inspection
Skim or skip:
Everything else - I don’t think there’s too much value to the details beyond the headline result, which is presented well in the intro.
Multimodal Neurons in Artificial Neural Networks (Gabriel Goh et al, OpenAI)
An analysis of neurons in a text + image model (CLIP), finding a bunch of abstract + cool neurons. Not a high priority to deeply engage with, but very cool and worth skimming.

My key takeaways
There are so many fascinating neurons! Like, what?
There’s a teenage neuron, a Minecraft neuron, a Hitler neuron and an incarcerated neuron?!
The intuition that multi-modal models (or at least, models that use language) are incentivised to represent things in a conceptual way, rather than specifically tied to the input format
The detailed analysis of the Donald Trump neuron, esp that it is more than just a “activates on Donald Trump” neuron, and instead activates for many different clusters of things, roughly tracking their association with Donald Trump.
This seems like weak evidence that neuron activations may split more into interpretable segments, rather than an interpretable directions
The “adversarial attacks by writing Ipod on an apple” part isn’t very deep, but is hilarious

Thanks to Trenton Bricken and Michael Nielson for nudging me to write an updated version!