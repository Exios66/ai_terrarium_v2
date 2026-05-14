# ai_terrarium_v2
# The AI Terrarium: Can LLMs Predict Voter Behavior?
**A Digital Twin Approach to Opinion Stimulation**
**Team:** Jack Burleson, Grant Mooslin, Dasey Dang
**Principal Investigators:** Tim Rogers, Dhavan Shah
**Context:** ICA 2026 • Computational Methods Division (10 Week Collaborative Project)
## 📌 Project Overview
This project explores a digital twin framework to answer a core research question: *If you were to give an LLM everything a real person told you about themselves (media diet, world view, economic anxieties, etc.), can it predict how they voted?*. Using ground truth data from the 2024 elections, this research tests the inferential capabilities of various Large Language Models (LLMs) to simulate public opinion and predict voter behavior based on structured persona prompts.
Present-day researchers utilize similar frameworks for public opinion and sentiment simulations; this project adds to the growing body of research clarifying the most effective prompting methods for replication and accurate outputs.
## 📊 Dataset & Demographics
The experiment relies on a robust dataset of real survey respondents, mirroring true demographic and political splits.
 * **Sample Size:** N = 1,768 respondents.
 * **Political Breakdown:** Approximately ⅓ Republican, ⅓ Democrat, and ⅓ Independent.
 * **Average Age:** 49 years.
 * **Data Sources:** Sampling frame built from ACS, voter records, CPS, NEP exit polls, and CES surveys.
The survey encompasses 366 questions divided into five primary insight categories:
 1. **Demographics (31 questions):** Age, race, religion, location, employment.
 2. **Media Diet (48 questions):** News outlets, social platforms, podcasts.
 3. **Worldview & Civic Attitudes (95 questions):** Economic perceptions, institutional trust, issue positions.
 4. **Political Behaviors (32 questions):** Participation, conversations, online activity.
 5. **Intelligence Ability (IQ) (4 items):** Verbal and non-verbal cognitive measures.
## ⚙️ Methodology & Experimental Workflow
### 1. Persona Prompt Generation
Raw survey variables (e.g., religpew → Roman Catholic, ideo5 → Conservative) were translated into natural language narratives that an LLM can reason about.
Two distinct prompting strategies were evaluated:
 * **Natural Language Prompting (Narrative Style):** A flowing, prose-based syntax written as ordinary discourse. This style prioritizes human readability and contextual nuance, embedding intent within sentence structure. Example: *"Religiously, you are Roman Catholic. You are registered to vote. Politically, you are conservative."*.
 * **Structured Survey Output (JSON Format):** Prompts encoded in JavaScript Object Notation, organizing context into explicit key-value pairs to prioritize machine readability and programmatic consistency.
### 2. Experimental Conditions (Context Depth)
Prompts were tested across five conditions of increasing informational depth:
 * **Condition 1:** Demographics only.
 * **Condition 2:** Demographics + Media Use.
 * **Condition 3:** Demographics + Media Use + Worldview.
 * **Condition 4:** Demographics + Media Use + Worldview + Political Behaviors.
 * **Condition 5:** Demographics + Media Use + Worldview + Political Behaviors + Intelligence Ability.
### 3. Inference & Evaluation
 * **Execution:** Inference prediction tasks were run through the Transformers package or VLLM via the Llama API. Output strings were then converted back to the participant sample format.
 * **Target Question (q21):** The models were tasked with answering a single ground-truth question: *"In the U.S. Presidential race, who do you plan to or did you vote for?"* (Options: Republican, Democrat, Someone else, Don't know).
 * **Evaluation:** Model outputs were computed for accuracy against this held-out question.
## 🤖 Models Tested
The experiment heavily leveraged Meta's open-source AI models alongside other lightweight instruct models:
 * **Llama-3.3-70b-Instruct:** The primary experimental model; a 70B parameter instruction-tuned generative model released December 6, 2024.
 * **Llama-3.0-7b-Instruct**
 * **Llama-3.1-8B-Instruct**
 * **Llama-3.2-3B-Instruct**
 * **Qwen2.5-0.5B-Instruct**
 * **SmolLM2-1.7B-Instruct**
## 📈 Key Findings & Results
Testing revealed that the **Llama-3.3-70B** model achieved the highest predictive performance when using Natural Language (Narrative) prompts, specifically peaking at **Condition 3**.
### Accuracy Breakdown (Llama-3.3-70B-Instruct)
| Prompt Format | Info Depth | Accuracy (Four Class) | Accuracy (Binary) |
|---|---|---|---|
| Natural Language | Condition 1 (Demo) | 0.7364 | 0.7872 |
| Natural Language | Condition 2 (Demo + Media) | 0.7602 | 0.8038 |
| **Natural Language** | **Condition 3 (Demo + Media + Worldview)** | **0.8281** | **0.8817** |
| Natural Language | Condition 4 (Cond 3 + Pol. Behaviors) | 0.8201 | 0.8817 |
| Natural Language | Condition 5 (Cond 4 + IQ) | 0.8179 | 0.8688 |
| Structured JSON | Condition 5 (All features) | 0.7494 | 0.7830 |
### The "Contextual Overload" Phenomenon
A major takeaway from the experimental results is that providing excessive information does not necessarily improve prediction accuracy. Moving from Condition 3 to Condition 5 (adding political behaviors and intelligence ability) actually led to a slight decline in accuracy (-0.010 for four-class and -0.013 for binary). This indicates that overly detailed or potentially noisy information may hinder an LLM's reasoning and performance.
Additionally, Structured JSON prompts consistently underperformed Natural Language narratives at equivalent informational depths.
## 🚧 Roadblocks & Limitations
Throughout the 10-week timeline, the team encountered challenges typical of real-life data science scenarios. These short-term roadblocks ranged from permission access errors to waiting on the completion of initial data collection.
## 🚀 Future Directions
To expand upon these initial findings, future phases of the *AI Terrarium* project will focus on:
 1. **Target Expansion:** Evaluating the models' ability to predict inferences on different held-out target questions, such as specific issue preferences, political participation rates, or media diet choices.
 2. **Model Diversification:** Replicating the experiment using different families of LLMs, including proprietary models from OpenAI and Anthropic, alongside other open-source alternatives.
 3. **Pipeline Optimization:** Improving and streamlining the data processing pipeline to allow for easier generalization and application to other survey samples.

