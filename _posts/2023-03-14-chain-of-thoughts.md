---
layout: post
title:  "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
date:   2023-03-14 15:26:24 +0100
categories: jekyll update
---
In this blog post, we will go over the [NeurIPS 2022 paper titled Chain-of-Thought Prompting Elicits Reasoning in Large Language Models][neurips-paper]. This paper introduces Chain of Thought (CoT) prompting, which augments few-shot training samples for in-context learning with explicit reasoning steps. The authors evaluate the effectiveness of CoT on math word problems, commonsense reasoning and symbolic reasoning and shows performance improvements from the standard prompting as well as fine-tuned GPT-3 with smaller number of parameters. In addition, a 540B-parameter language model PaLM prompted with CoT outperforms fine-tuned 175B GPT-3.

# Outline of this blog post:

1. [Introduction](#introduction)  
  1.1. [Motivation behind chain-of-thought prompting](#1.1)  
  1.2. [What is Chain of Thought Prompting?](#1.2)  
  1.3. [Attractive Properties of CoT Prompting](#1.3)  
2. [Experiments](#experiments)  
  2.1. [Arithmetic Reasoning](#2.1)  
  &emsp;2.1.1. [Benchmarks & Results](#2.1.1)  
  &emsp;2.1.2. [Ablation Study](#2.1.2)  
  &emsp;2.1.3. [Robustness of Chain of Thought](#2.1.3)  
  2.2. [Commonsense Reasoning](#2.2)   
  &emsp;2.2.1. [Benchmarks & Results](#2.2.1)  
  2.3. [Symbolic Reasoning](#2.3)  
	&emsp;2.3.1. [Tasks & Results](#2.3.1)  
3. [Conclusions](#conclusion)  
4. [References](#references)  


## 1. Introduction <a name="introduction"></a>

# 1.1 Motivation behind chain-of-thought prompting <a name="1.1"></a>

Language models have transformed the field of natural language processing by improving performance and sample efficiency. However, increasing the size of models alone does not guarantee high performance on challenging tasks such as arithmetic, commonsense, and symbolic reasoning. This paper proposes a method called "chain-of-thought prompting" which is inspired by several prior directions: prompting, natural language explanations, program synthesis/execution, numeric and logical reasoning, and intermediate language steps. The recent success of large-scale language models has led to growing interest in improving their capability to perform tasks via prompting (Brown et al., 2020).  
CoT prompting combines two ideas to unlock the reasoning ability of large language models:

**Firstly, generating natural language rationales can improve arithmetic reasoning.**

Prior work has generated natural language intermediate steps by training from scratch (Ling et al., 2017) or finetuning a pretrained model (Cobbe et al., 2021), as well as neuro-symbolic methods that use formal languages instead of natural language (Roy and Roth, 2015; Chiang and Chen, 2019; Amini et al., 2019; Chen et al., 2019). 

**Secondly, large language models can perform in-context few-shot learning via prompting**, which means that instead of finetuning a separate language model for each new task, one can simply “prompt” the model with a few input–output exemplars demonstrating the task.

Empirical evaluations on arithmetic, commonsense, and symbolic reasoning benchmarks demonstrate that chain-of-thought prompting outperforms standard prompting. This approach is important because it does not require a large training dataset and a single model checkpoint can perform many tasks without loss of generality.

<figure>
    <center><img src="/assets/images/1.png"
         alt="figure 1">
    <figcaption><font color="grey">CoT increases the range of tasks that big LMs can do.<a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202022/54087.png?t=1669146827.101349" target="_blank"> Image Source.</a></font></figcaption></center>
</figure>


# 1.2. What is Chain-of-Thought Prompting anyway? <a name="1.2"></a>

A chain of thought is a series of intermediate natural language reasoning steps that lead to the final output, and this approach is called **chain-of-thought prompting**. An example prompt is shown in the figure below. 

<figure>
    <center><img src="/assets/images/2.png"
         alt="figure 2"></center>
</figure>

The first method, called "standard prompting" (<a href="https://arxiv.org/abs/2005.14165" target="_blank">popularized by GPT-3</a>), involves providing the model with input-output pairs (questions and answers) before asking it to predict the answer for a test example. The second method, called "chain of thought prompting," involves prompting the model to generate intermediate reasoning steps before giving the final answer to a problem. The goal is to simulate an intuitive thought process that humans might use when working through a multi-step reasoning problem. While <a href="https://arxiv.org/abs/2006.06609" target="_blank"> previous methods</a> have used fine-tuning to produce such thought processes, the authors show that chain of thought prompting can elicit these processes by providing a few examples of the chain of thought without requiring a large training dataset or modifying the language model's weights.

# 1.3. Attractive Properties of CoT Prompting <a name="1.3"></a>

<figure>
    <center><img src="/assets/images/3.png"
         alt="figure 3">
    <figcaption><font color="grey">Several reasons why CoT is good.<a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202022/54087.png?t=1669146827.101349" target="_blank"> Image Source.</a></font></figcaption></center>
</figure>

The approach of chain-of-thought prompting has several advantages for enabling reasoning in language models:  
<ul><li>It allows models to break down multi-step problems into intermediate steps, which enables them to allocate additional computation for problems that require more reasoning steps.</li><li>A chain of thought offers an interpretable view of the model's behavior, indicating how it might have reached a specific answer and allowing for debugging if the reasoning process goes wrong.</li><li>Chain-of-thought reasoning can be used for a range of tasks like math word problems, commonsense reasoning, and symbolic manipulation, and can potentially be applied to any task that humans solve via language.</li><li>It can be quickly elicited in large off-the-shelf language models by providing examples of chain of thought sequences as exemplars for few-shot prompting.</li></ul>
Empirical experiments that will be discussed below show the effectiveness of chain-of-thought prompting for arithmetic reasoning (Section 2.1 <a name="2.1"></a>), commonsense reasoning (Section 2.2 <a name="2.2"></a>), and symbolic reasoning (Section 2.3 <a name="2.3"></a>).

## 2. Experiments <a name="experiments"></a>

# 2.1. Arithmetic reasoning <a name="2.1"></a>
Let’s start with math word problems of the form in the above figure, which measure the arithmetic reasoning ability of language models. Though simple for humans, arithmetic reasoning is a task where language models often struggle to use logical deduction and problem-solving skills. In the context of natural language processing and artificial intelligence, arithmetic reasoning tasks are used to evaluate the ability of language models to perform mathematical calculations and solve mathematical word problems.

# 2.1.1 Benchmarks & Results <a name="2.1.1"></a>

Chain-of-thought prompting is here explored for various language models on multiple benchmarks.
Five math word problem benchmarks include: (1) the <a href="https://arxiv.org/abs/2110.14168" target="_blank">GSM8K</a> benchmark of math word problems, (2) the <a href="https://aclanthology.org/2021.naacl-main.168/" target="_blank">SVAMP</a> dataset of math word problems with varying structures, (3) the <a href="https://aclanthology.org/2020.acl-main.92/" target="_blank">ASDiv</a> dataset of diverse math word problems, (4) the <a href="https://arxiv.org/pdf/1705.04146.pdf" target="_blank">AQuA</a> dataset of algebraic word problems, and (5) the <a href="https://aclanthology.org/N16-1136.pdf" target="_blank">MAWPS</a> benchmark. 

<figure>
    <center><img src="/assets/images/4.png"
         alt="figure 4">
    <figcaption><font color="grey">Example questions for math word problems.</font></figcaption></center>
</figure>

<figure>
    <center><img src="/assets/images/5.png"
         alt="figure 5">
    <figcaption>
    <font color="grey">Summary of math word problem benchmarks used in the paper with examples. N: number of evaluation examples.</font>
    </figcaption></center>
</figure>

**Standard prompting.** For the baseline, we consider <a href="https://arxiv.org/abs/2005.14165" target="_blank">standard few-shot prompting</a>, in which a language model is given in-context exemplars of input–output pairs before outputting a prediction for a test-time example. Exemplars are formatted as questions and answers. The model gives the answer directly, as shown in the above figure (left).

**CoT Prompting.** The proposed method involves adding a chain of thought for each example in the few-shot prompting dataset. The researchers manually created eight few-shot examples with chains of thought for prompting since most datasets only have an evaluation split (first figure, right). To test the effectiveness of chain-of-thought prompting for various math word problems, the same set of eight chain of thought exemplars was used for all benchmarks except AQuA, which is multiple-choice and uses four exemplars and solutions from the training set.
The two tables below show few-shot exemplars for CoT prompting for all datasets.

<figure>
    <center><img src="/assets/images/7.png"
         alt="figure 7">
    </center>
</figure>

<figure>
    <center><img src="/assets/images/8.png"
         alt="figure 8">
    </center>
</figure>

The task is performed by five large language models. The first is <a href="https://arxiv.org/abs/2005.14165" target="_blank">GPT-3</a>, for which text-ada-001, text-babbage-001, text-curie-001, and text-davinci-002 models are used, which presumably correspond to <a href="https://arxiv.org/abs/2203.02155" target="_blank">InstructGPT</a>models of 350M, 1.3B, 6.7B, and 175B parameters. The second is <a href="https://arxiv.org/abs/2201.08239" target="_blank">LaMDA</a>, which has models of 422M, 2B, 8B, 68B, and 137B parameters. The third is <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html" target="_blank">PaLM</a>, which has models of 8B, 62B, and 540B parameters. The fourth is <a href="https://arxiv.org/abs/2205.05131" target="_blank">UL2</a> 20B, and the fifth is <a href="https://arxiv.org/abs/2107.03374" target="_blank">Codex</a>. They sample from the models via <a href="https://medium.com/nlplanet/two-minutes-nlp-most-used-decoding-methods-for-language-models-9d44b2375612" target="_blank">greedy decoding</a>(though <a href="https://arxiv.org/abs/2203.11171" target="_blank">follow-up work</a> shows chain-of-thought prompting can be improved by taking the majority final answer over many sampled generations). 

**Results.** All experimental outputs for each model collection, model size, and benchmark are shown in the table below.

<figure>
    <center><img src="/assets/images/9.png"
         alt="figure 9">
    </center>
</figure>

The strongest results of chain-of-thought prompting are summarized in the figure below. There are three key takeaways:  
1. CoT prompting does not improve performance for small models and is only effective for models of ~ 100 B parameters. Smaller models produced fluent but illogical chains of thought, resulting in lower performance than standard prompting.
2. CoT prompting has larger performance gains for more complicated problems. It doubled performance for the largest GPT and PaLM models on the GSM8K dataset, which had the lowest baseline performance. However, it had negative or very small improvements for the SingleOp subset of MAWPS, which only requires a single step to solve.
3. CoT prompting using GPT-3 175B and PaLM 540B compares favorably to prior state of the art, which usually involves fine-tuning a task-specific model on a labeled training dataset. PaLM 540B achieves new state of the art on GSM8K, SVAMP, and MAWPS using chain-of-thought prompting.

<figure>
    <center><img src="/assets/images/10.png"
         alt="figure 10">
    </center>
</figure>

To understand why chain-of-thought prompting works, a manual examination of LaMDA 137B model-generated chains of thought for GSM8K was done. The analysis found that 50 randomly selected examples where the model returned the correct final answer, all of the generated chains of thought were logically and mathematically correct except for two coincidences. The analysis also examined 50 random samples where the model gave the wrong answer, finding that 46% of the chains of thought were almost correct, with minor mistakes, and that 54% had major errors in semantic understanding or coherence. A similar analysis of errors made by PaLM 62B was performed and it showed that scaling to PaLM 540B improved its one-step missing and semantic understanding errors.

## 2.1.2. Ablation Study <a name="2.1.2"></a>

Given the observed benefits of using chain-of-thought prompting, can we say that the same performance improvements can be conferred via other types of prompting? To answer this question, let’s look at the ablation study shown in the figure below with three variations of chain of thought.

<figure>
    <center><img src="/assets/images/11.png"
         alt="figure 11">
         <figcaption><font color="grey">Ablation study for different variations of prompting using LaMDA 137B & PaLM 540B.</font></figcaption>
    </center>
</figure>

**Equation only.** The first variation prompts the model to output only a mathematical equation before giving the answer, but Figure 5 shows that this does not help much for the GSM8K dataset, implying that the semantics of the questions in GSM8K are too challenging to directly translate into an equation without the natural language reasoning steps.  
**Variable compute only.** The second variation isolates the effect of variable computation from chain-of-thought reasoning by prompting the model to output only a sequence of dots equal to the number of characters in the equation needed to solve the problem, but this performs about the same as the baseline, suggesting that variable computation by itself is not the reason for the success of chain-of-thought prompting.   
**Chain of thought after answer.** The third variation tests whether the model actually depends on the produced chain of thought to give the final answer by giving the chain of thought prompt only after the answer, but this performs about the same as the baseline, suggesting that the sequential reasoning embodied in the chain of thought is useful for reasons beyond just activating knowledge.

## 2.1.3. Robustness of Chain of Thought <a name="2.1.3"></a>

The robustness of CoT prompting is evaluated by different chains of thought written by different annotators. Three co-authors of the paper independently wrote chains of thought for the same few-shot exemplars, and a comparison of the results for LaMDA 137B on GSM8K and MAWPS in the figure below shows that all sets of CoT prompts outperform the standard baseline by a large margin, implying that successful use of CoT does not depend on a particular linguistic style. Additionally, experiments with three sets of eight exemplars randomly sampled from the GSM8K training set show that these prompts performed comparably with the manually written exemplars, also substantially outperforming standard prompting. Finally, the study finds that CoT prompting for arithmetic reasoning is robust to different exemplar orders and varying numbers of exemplars.

<figure>
    <center><img src="/assets/images/12.png"
         alt="figure 12">
    </center>
</figure>

# 2.2. Commonsense reasoning <a name="2.2"></a>

In addition to arithmetic reasoning, we consider whether the language-based nature of chain of thought prompting also makes it applicable to commonsense reasoning, which involves reasoning about physical and human interactions under the presumption of general background knowledge. 

# 2.2.1. Benchmarks & Results <a name="2.2.1"></a>

For these evaluations, we observe the <a href="https://aclanthology.org/N19-1421" target="_blank">CommonsenseQA</a> and <a href="https://aclanthology.org/2021.tacl-1.21" target="_blank">StrategyQA</a> benchmarks, as well as two domain-specific tasks from <a href="https://github.com/google/BIG-bench/" target="_blank">BIG-Bench collaboration</a> regarding <a href="https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/date_understanding" target="_blank">date understanding</a> and <a href="https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/sports_understanding" target="_blank">sports understanding</a>. In addition, the <a href="https://say-can.github.io/" target="_blank">SayCan</a> dataset involves mapping a natural language instruction to a sequence of robot actions from a discrete set.

<figure>
    <center><img src="/assets/images/13.png"
         alt="figure 13">
    <figcaption><font color="grey">Example questions for commonsense reasoning. Source: <a href="https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html" target="_blank"> Google Research.</a></font></figcaption></center>
</figure>

We follow the same experimental setup for prompts as the prior section.

**Results.** The full results for LaMDA, GPT-3, and different model scales are shown in the table below.

<figure>
    <center><img src="/assets/images/14.png"
         alt="figure 14">
    </center>
</figure>

The figure below shows that PaLM of a bigger size performed best in all tasks, indicating the gain of CoT prompting. PaLM 540B achieved higher improvement compared to baselines and exceeded the prior state of the art on StrategyQA (75.6% vs 69.4%) and outperformed sports enthusiasts on sports understanding (95.4% vs 84%).

<figure>
    <center><img src="/assets/images/15.png"
         alt="figure 15">
    </center>
</figure>

# 2.3. Symbolic reasoning <a name="2.3"></a>

Our final experimental evaluation considers symbolic reasoning, which is simple for humans but potentially challenging for language models.

# 2.3.1. Tasks & Results <a name="2.3.1"></a>

We use the following two toy tasks:  
**1. Last letter concatenation.** This task asks the model to concatenate the last letters of words in a name. It is a more challenging version of first letter concatenation, which language models can already perform without chain of thought.  
**2. Coin flip.** This task asks the model to answer whether a coin is still heads up after people either flip or don’t flip the coin.

<figure>
    <center><img src="/assets/images/16.png"
         alt="figure 16">
         <figcaption><font color="grey">Example questions for symbolic reasoning.</font></figcaption>
    </center>
</figure>

The performance of language models on symbolic reasoning tasks is evaluated using in-domain and out-of-domain test sets for which evaluation examples had more steps than those in the few-shot exemplars. For last letter concatenation, the model only sees exemplars of names with two words, and then performs last letter concatenation on names with 3 and 4 words. 4 We do the same for the number of potential flips in the coin flip task. Our experimental setup uses the same methods and models as in the prior two sections.

**Results.** As shown in the table below, with PaLM 540B, CoT prompting leads to almost 100% solve rates. It is worth to note that standard prompting already solves coin flip with PaLM 540, but not for LaMDA 137B. 

<figure>
    <center><img src="/assets/images/17.png"
         alt="figure 17">
    </center>
</figure>

Overall results demonstrate that small models still fail and the ability to perform abstract manipulations on unseen symbols for these three tasks only arises at the scale of 100B model parameters. As for the out-of-domain evaluations, standard prompting fails for both tasks. With CoT prompting, language models achieve better results, although performance is lower than in the in-domain setting. Hence, chain-of-thought prompting facilitates symbolic reasoning beyond seen chains of thought for language models of sufficient scale.

## 3. Conclusions <a name="conclusion"></a>

In this blog post, we provided a better understanding of chain-of-thought prompting for multi-step reasoning behavior in large language models. The experiments of the paper <em>Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</em> showed that CoT prompting improves performance on arithmetic, commonsense, and symbolic reasoning tasks, and allows large language models to perform tasks that they would otherwise be unable to. Although CoT prompting has its own limitations such as the cost of manual annotation and the lack of guarantee of correct reasoning paths, the paper suggests that this is a simple and broadly applicable method for enhancing reasoning in language models, and may inspire further work on language-based approaches to reasoning.

## 4. References <a name="references"></a>

<a href="https://arxiv.org/abs/2204.01691" target="_blank">Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al. 2022. Do as I can, not as I say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691.</a>  

<a href="https://aclanthology.org/N19-1245" target="_blank">Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota. Association for Computational Linguistics.</a>  

<a href="https://github.com/google/BIG-bench/" target="_blank">BIG-bench collaboration. 2021. Beyond the imitation game: Measuring and extrapolating the capabilities of language models. In preparation.</a>   

<a href="https://arxiv.org/pdf/2005.14165.pdf" target="_blank">Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. NeurIPS.</a>   

<a href="https://openreview.net/forum?id=ryxjnREFwH" target="_blank">Xinyun Chen, Chen Liang, Adams Wei Yu, Denny Zhou, Dawn Song, and Quoc V. Le. 2019. Neural symbolic reader: Scalable integration of distributed and symbolic representations for reading comprehension. ICLR.</a>  

<a href="https://arxiv.org/abs/2107.03374" target="_blank">Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.</a>  

<a href="https://doi.org/10.18653/v1/N19-1272" target="_blank">Ting-Rui Chiang and Yun-Nung Chen. 2019. Semantically-aligned equation generation for solving and reasoning math word problems. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2656–2668, Minneapolis, Minnesota. Association for Computational Linguistics.</a>  

<a href="https://medium.com/nlplanet/two-minutes-nlp-most-used-decoding-methods-for-language-models-9d44b2375612" target="_blank">Fabio Chiusano. 2022. Two minutes NLP — Most used Decoding Methods for Language Models. NLPlanet.</a>  

<a href="https://arxiv.org/pdf/2110.14168.pdf" target="_blank">Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.</a>  

<a href="https://doi.org/10.1162/tacl_a_00370" target="_blank">Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. TACL.</a>  

<a href="https://doi.org/10.18653/v1/N16-1136" target="_blank">Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: A math word problem repository. NAACL.</a>  

<a href="https://aclanthology.org/P17-1015.pdf" target="_blank">Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. ACL.</a>  

<a href="https://doi.org/10.18653/v1/2020.acl-main.92" target="_blank">Shen Yun Miao, Chao Chun Liang, and Keh Yih Su. 2020. A diverse corpus for evaluating and developing English math word problem solvers. ACL.</a>  

<a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html" target="_blank">Sharan Narang, Aakanksha Chowdhery. 2022. Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance. Google research.</a>  

<a href="https://arxiv.org/abs/2203.02155" target="_blank">Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.</a>  

<a href="https://aclanthology.org/2021.naacl-main.168.pdf" target="_blank">Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? NAACL.</a>  

<a href="https://aclanthology.org/D15-1202.pdf" target="_blank">Subhro Roy and Dan Roth. 2015. Solving general arithmetic word problems. EMNLP</a>  

<a href="https://doi.org/10.18653/v1/N19-1421" target="_blank">Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. NAACL.</a>  

<a href="https://arxiv.org/abs/2205.05131" target="_blank">Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, and Donald Metzler. 2022. Unifying language learning paradigms. arXiv preprint arXiv:2205.05131.</a>  

<a href="https://arxiv.org/abs/2006.06609" target="_blank">Alon Talmor, Oyvind Tafjord, Peter Clark, Yoav Goldberg, and Jonathan Berant. 2020. Leap-of-thought: Teaching pre-trained models to systematically reason over implicit knowledge. NeurIPS.</a>  

<a href="https://arxiv.org/abs/2201.08239" target="_blank">Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. LaMDA: Language models for dialog applications. arXiv preprint arXiv:2201.08239.</a> 

<a href="https://arxiv.org/abs/2203.11171" target="_blank">Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. 2022a. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.</a>  

<a href="https://arxiv.org/pdf/2201.11903.pdf" target="_blank">Jason Wei, Wang Xuezhi, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.</a>   

<a href="https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html" target="_blank">Jason Wei, Denny Zhou. 2022. Language Models Perform Reasoning via Chain of Thought. Google research.</a>   

[neurips-paper]: https://arxiv.org/pdf/2201.11903.pdf

