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
This research project intends to explore the potential of a Socratic Model (SM) pipeline, incorporating a Vision Language Model (VLM) and a Language Model (LM), to solve logical puzzles represented as images, such as the [Raven Progressive Matrices](https://wellyzhang.github.io/project/raven.html) (RPM). The experiment will focus on assessing the general reasoning capabilities of the language model in the proposed pipeline and compare the results to baselines, such as GPT4 and mini-GPT4. The research objectives are to assess the performance of a Socratic model on logical puzzles, compare it to existing methods, and explore its potential to extend to other applications.

The significance of the proposed work is that it will provide insight into the capabilities of a Socratic model to solve logical puzzles, as well as its potential to extend to new applications. It will also provide a comparison between existing models and the proposed one.

The proposed work consists of constructing a pipeline using the Socratic model framework and using an open-source implementation of different VLMs and LMs, such as Clip, Open-Assistant, Codex and FlanT5. Also traditional Computer Vision methods will be tested to extract the shapes of the Raven Progressive Matrices. The exact models are determined by testing the performance of different models. The performance of this pipeline will be assessed using a metric based on the outcomes of the logical puzzles, and the results will be compared to the baselines.

The anticipated outcomes from this project include insight into the performance of the Socratic model pipeline and the potential for it to extend to other applications. Additionally, it is expected that this work will provide a comparison between the proposed pipeline and existing models in the field.

## Experiments
![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/a2f98ac5-a13c-4349-be16-dda4ee6bc6b4)

With our experiment we test the Socratic Model on the Center Single Raven Progressive Matrices. These include simple shapes such as triangles, squares, circles, hexagon and pentagons. The final shape is unknown and should be predicted based on the pattern of the other shapes. An example:


![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/824a5d77-09a1-4e4c-8ebc-12afa098c2ca)


### Experiment 1
In this experiment different LLMs are tested to predict the answer of the RPM, based on the groundtruth shape, color, size and angle of the images. This experiment is done to assess the capibilities of the Langauge Model to solve the RPM with perfect information.

### Experiment 2
In this experiment the LLMs are connected to different VLMs to test the performance of the complete Socratic Model pipeline. 

### Experiment 3
![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/ad25e324-e62c-4573-9cd8-70008552dc9a)

In the final experiment the performance of the Socratic Model is compared to [Flamingo](https://arxiv.org/pdf/2204.14198.pdf). Flamingo is a Multi Modal Visual Language Model for few shot learning. It uses gates to constrain the Language Model with the encoded Vision input. In our case this is the puzzle, and the different shapes of the answer. This model is trained for Visual Question Answering (VQA) and Few shot learning. Because Flamingo is not opensource, we use the [OpenFlamingo version](https://github.com/mlfoundations/open_flamingo) with LLama as Large Language Model. To compare it to other methods, we also compare it to the methods proposed by the original authors of the [Raven Dataset](https://wellyzhang.github.io/attach/cvpr19zhang.pdf). This is a method with a ResNET backbone and Dynamic Residual Tree. This method is not retested with our research. They also compare it to other Deep Learning methods, which will be shown in the conclusion paragraph.

# Results
Although we have not gathered all the necessary results, early testing shows promising results on the centre_single (an RPM with only one shape in the centre of the image) seems to provide excellent results with around 80-90% accuracy. Limited testing on the more complicated sets shows that the SM has more difficulty when there isn’t a single shape to focus on. More concrete results, both qualitative and quantitative, will follow once we run the full suite of experiments.

### Experiment 3
| **Our method**|                        |                           |          |
|---------------|------------------------|---------------------------|----------|
| Model         | Macro average F1 score | Weighted average F1 score | Accuracy |
| CLIP + Flant5 |                        |                           |          |
| OpenFlamingo  | 0.05                   | 0.05                      | 0.13     |


# Conclusion
| **Our method**|                        |                           |          |
|---------------|------------------------|---------------------------|----------|
| Model         | Macro average F1 score | Weighted average F1 score | Accuracy |
| CLIP + Flant5 |                        |                           |          |
| OpenFlamingo  | 0.05                   | 0.05                      | 0.13     |
| **Other method**                       |                           |          |
| LSTM          |                        |                           | 0.13     |
| WReN          |                        |                           | 0.13     |
| CNN           |                        |                           | 0.34     |
| ResNET        |                        |                           | 0.52     |
| LSTM+DRT      |                        |                           | 0.14     |
| WReN +DRT     |                        |                           | 0.15     |
| CNN +DRT      |                        |                           | 0.37     |
| ResNET +DRT   |                        |                           | 0.58     |
| Human         |                        |                           | 0.95     |
| Solver        |                        |                           | 1.00     |
