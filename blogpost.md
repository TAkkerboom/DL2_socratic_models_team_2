# Introduction
In Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [[1]](https://arxiv.org/pdf/2204.00598), the authors introduce Socratic Models (SMs). SMs enable the composition of multiple pre-trained models in a modular framework, allowing them to exchange information with one another and capture new multimodal capabilities without the need for fine tuning. These models are based on large pre-trained models and exhibit distinct capabilities depending on the domain of data they are trained on. By combining these models, SMs have been shown to be competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval. Moreover, SMs can additionally be extended using non-neural modules, such as a web-crawler. As such, SMs open up a range of new applications including answering free-form questions about the egocentric video, engaging in multimodal assistive dialogue with people by interfacing with external APIs and databases, and robot perception and planning.

Research on Socratic Models has been ongoing for several years and has included a range of approaches and experiments. Examples of common techniques used in the field include Natural Language Processing (NLP) for text processing, Computer Vision (CV) for image processing, and Reinforcement Learning (RL) for model optimization. Progress in the field has been made through experiments such as zero-shot image captioning, video-to-text retrieval, and multimodal assistive dialogue. The current status of this domain is that Socratic Models are competitive with state-of-the-art methods in these areas, and they offer the potential to extend to new applications.


# Strengths/Weaknesses
## Strengths:
No training/finetuning required. 
The idea behind Socratic Models is that we can chain existing models as is, without any training. This eliminates the need for additional compute resources and time-consuming training processes. By leveraging pre-trained models, Socratic Models can quickly provide intelligent and context-aware responses. This approach also ensures that the system remains up-to-date with the latest information and developments, as it operates based on existing knowledge. Additionally, when the state-of-the-art advances, modules can quickly be swapped. The absence of training requirements allows for effortless integration and implementation of Socratic Models across various applications and platforms, making it a versatile and efficient solution for natural language understanding and generation tasks.


Multimodality and versatility
Due to chaining of the VLMs with the LMs, Socratic Models have multimodal understanding. Due to the multimodal understanding, the Socratic Models can be applied to a wide range of use cases. 


Better language understanding
Because VLMs are trained with contrastive learning, they don’t have an understanding of language. Chaining LMs with VLMs will improve the understanding of language, which gives an improvement of the total model, in comparison with only VLMs. 

## Weaknesses:
Error propagation.
One potential challenge in Socratic Models is the risk of error propagation. If the initial language model provides an incorrect or biassed response, subsequent models in the chain may continue to build upon that flawed information, leading to a cascade of inaccurate or misleading answers. Addressing error propagation requires careful monitoring, evaluation, and periodic intervention to correct any erroneous responses and maintain the overall quality and reliability of the system.


No common embedding space between models.
Not having a common embedding space in Socratic Models can pose challenges in seamlessly integrating and transferring knowledge between different language models. Without a shared representation, it becomes more difficult to leverage the strengths of individual models and to ensure consistent understanding and coherent responses across the chain. This lack of a common embedding space may result in inconsistencies, conflicting information, and difficulties in maintaining a cohesive and coherent conversational flow.


Ambiguity of language as the medium.
Since we use language as the medium in Socratic Models there is the potential for ambiguity and misinterpretation. 

# Contribution
This research project intends to explore the potential of a Socratic Model (SM) pipeline, incorporating a Vision Language Model (VLM) and a Language Model (LM), to solve logical puzzles represented as images, such as the Raven Progressive Matrices (https://wellyzhang.github.io/project/raven.html). The experiment will focus on assessing the general reasoning capabilities of the language model in the proposed pipeline and compare the results to baselines, such as GPT4 and mini-GPT4. The research objectives are to assess the performance of a Socratic model on logical puzzles, compare it to existing methods, and explore its potential to extend to other applications.

The significance of the proposed work is that it will provide insight into the capabilities of a Socratic model to solve logical puzzles, as well as its potential to extend to new applications. It will also provide a comparison between existing models and the proposed one.

The proposed work consists of constructing a pipeline using the Socratic model framework and using an open-source implementation of different VLMs and LMs, such as Flamingo, Clip, Open-Assistant, Codex and FlanT5. Also traditional Computer Vision methods will be tested to extract the shapes of the Raven Progressive Matrices. The exact models are determined by testing the performance of different models. The performance of this pipeline will be assessed using a metric based on the outcomes of the logical puzzles, and the results will be compared to the baselines.

The anticipated outcomes from this project include insight into the performance of the Socratic model pipeline and the potential for it to extend to other applications. Additionally, it is expected that this work will provide a comparison between the proposed pipeline and existing models in the field.


# Results
Although we have not gathered all the necessary results, early testing shows promising results on the centre_single (an RPM with only one shape in the centre of the image) seems to provide excellent results with around 80-90% accuracy. Limited testing on the more complicated sets shows that the SM has more difficulty when there isn’t a single shape to focus on. More concrete results, both qualitative and quantitative, will follow once we run the full suite of experiments.

# Conclusion
TBD
