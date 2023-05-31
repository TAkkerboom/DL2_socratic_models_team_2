# Solving Visual-Language Logic Puzzles with Socratic Models
28/05/2023 | Sergei Agaronian, Theodoor Akkerboom, Maarten Drinhuyzen, Wenkai Pan, Marius Strampel

## Introduction
Zeng et al. (2022) [[1]](#sm) introduce Socratic
Models (SMs). SMs enable the composition of multiple large, pre-trained models in a modular framework, allowing them to exchange information with one another and to tackle new multimodal challenges without the need for fine-tuning. These models are based on large pretrained models and can vary their capabilities wildly depending on the domain they are trained on. By combining such models, SMs have been shown to be competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval. Moreover, SMs can additionally be extended using non-neural modules, such as a web-crawler. As such, SMs open up a range of new applications including answering free-form questions about the egocentric video, engaging in multimodal assistive dialogue with people by interfacing with external APIs and databases, and robot perception and planning.

This research project aims to investigate the ability of a SM pipeline to solve logical puzzles represented as images, specifically on Raven Progressive Matrices (RPMs). Our proposed pipeline consists of a Visual Language Models (VLM) and a Large Language Model (LLM). The primary objective of the experiments is to evaluate the general reasoning capabilities of the language model within the proposed pipeline and compare its performance against baselines, including the standalone language model and Google DeepMind's Flamingo.

## Socratic Models
### Background
The core concept of SMs is to harness what Zeng et al. (2022) call differences in common sense knowledge. Visual Language Models (VLMs), for example, are trained on image captions, and thus develop understanding of the connection between images and language. Language Models (LMs) are trained on additional corpora that do not need to possess a visual aspect, such as novels or recipes. As such, they are applicable to a wide variety of linguistic tasks. By using SMs, we can harness these complementary differences in knowledge and apply them to both existing and new tasks out-of-the-box.

![image](https://gyazo.com/8d2f8d1a893ed836f6c9dc12ef927753.png)

Figure 1. Source: [[1]](#sm). Differing and complementary commonsense knowledge learned by different foundation models of differing modalities.

Furthermore, SMs are designed to facilitate human-like reasoning and interpretation. They aim to emulate the Socratic method, a philosophical approach that involves asking probing questions to stimulate critical thinking and uncover deeper insights. SMs employ similar techniques by generating questions, seeking clarifications, and engaging in interactive dialogue with users. This approach enhances the model's ability to understand complex concepts, handle ambiguity, and provide more nuanced responses.

Overall, Socratic Models combine the complementary knowledge of VLMs and LMs with the power of Socratic questioning to generate more insightful and reasoned responses. By fostering critical thinking and adaptability, SMs have the potential to revolutionize various domains, from education and research to decision-making and problem-solving in a diverse range of applications.

### Strengths and Weaknesses
The idea behind Socratic Models is to chain existing models as is, without any training or fine-tuning. This eliminates the need for additional compute resources and time-consuming training processes. By leveraging pre-trained models, Socratic Models can quickly provide intelligent and context-aware responses. This approach also ensures that the system remains up-to-date with the latest information and developments, as it operates based on existing state-of-the-art. In addition, as the state of the art advances, modules can be quickly swapped. The absence of training requirements allows for effortless integration and implementation of Socratic Models across various applications and platforms, making it a versatile and efficient solution for natural language understanding and generation tasks.

Additionally, SM's modular design allows chaining of models with different modalities, such as VLMs with LMs. This allows SMs to have multimodal understanding, greatly increasing the range of problems SMs can tackle. Research by Zeng et al. (2022) demonstrates that SMs can thus be competitive with state-of-the-art on image captioning, video-to-text retrieval, and more.

It is worth noting that VLMs, trained using contrastive learning, may not possess the same level of language understanding as LMs. By chaining LMs with VLMs, the overall model's language comprehension is enhanced, resulting in improvements over using VLMs alone.

However, chaining multiple models in Socratic Models introduces the possibility of error propagation through its constituent components. If the initial language model provides an incorrect or biassed response, subsequent models in the chain may continue to build upon that flawed information, leading to a cascade of inaccurate or misleading answers. Addressing error propagation requires careful monitoring, evaluation, and periodic intervention to correct any erroneous responses and maintain the overall quality and reliability of the system.

Another potential challenge lies in the lack of a shared embedding space among the foundation models used in Socratic Models. Without a common embedding space, seamless transfer of knowledge between models becomes difficult, and language acts as the medium for their interaction. This can pose a problem if the language understanding of a model is limited, either in general or domain-specific, as information might get lost in translation. Without a shared representation, it becomes more difficult to leverage the strengths of individual models and to ensure consistent understanding and coherent responses across the chain. This lack of a common embedding space may result in inconsistencies, conflicting information, and difficulties in maintaining a cohesive and coherent conversational flow. All of this is on top of language itself often being ambiguous.

## Contribution
In this research, we investigate the logical reasoning capabilities of Socratic Models. To achieve this, we use Raven's Progressive Matrices puzzles, which will be elaborated on later in the report. The main objective of the experiments is to evaluate the general reasoning abilities of the Socratic Models within the proposed pipeline and compare its performance against baselines, such as the standalone language model and Google DeepMind's Flamingo.

Our proposed work aims to construct a pipeline for the SM framework, incorporating state-of-the-art Visual Language Models (VLMs) and Language Models (LMs) namely Clip, Blip, and FlanT5, which are open source models. In total, we conduct three experiments: the first one focuses on the language model alone, the second one evaluates the combination of the language model with a visual language model, and the third one evaluates Flamingo, an integration of both models. 

Furthermore, we explore traditional Computer Vision methods to extract the shapes from the Raven Progressive Matrices. The selection of specific models is based on rigorous performance tests. To assess the effectiveness of this pipeline, a comprehensive evaluation is conducted, which includes both quantitative and qualitative analyses. The quantitative results are compared against established baselines.

### Raven Progressive Matrices
For all experiments, the pipeline is tested on the Center Single subset of the Raven Progressive Matrices dataset by Zhang et al. (2019) [[2]](#raven). An RPM consists of a three-by-three grid of items, with the final item missing. There are eight potential solutions of which the model must pick the missing shape to finish the pattern. An example is shown in Figure 2 below. This study aims to provide a proof-of-concept for solving logical puzzles such as RPMs with SMs and thus sticks to the Centre Single subset to avoid logistic difficulties. Each item contains a single shape (triangle, square, circle, hexagon, or pentagon). The correct answer is identified by the shape itself and its additional attributes, namely size, colour, and angle. The Centre Single subset for testing contains 2000 puzzles. Additional subsets exist, namely with multiple shapes and even overlapping shapes in each of the sections/answers, however, we do not tackle them in this research. 

<br>![image](https://gyazo.com/c6d0bcea47b740917a436c0f8ab3411c.png)

Figure 2. RPM Examples. The top 8 figures with the missing shape form a logical puzzles, where the missing shape should be filled with one of the 8 answers below. Only one answer can be correct at a time.

### Baselines
The baseline we use to compare our pipeline is from the paper that originated the RPM dataset [[2]](#raven). In the paper, they used a Dynamically Residual Tree to solve the puzzles. The Dynamically Residual Tree is a tree traversal algorithm which solves the puzzle by going through the nodes. To get the visual encoding of the shapes, they used ResNET. After training, their algorithm was compared to human performance and a logical solver algorithm.

Table 1. Baseline for comparison

| Model| accuracy   |
|------|-------------|
| ResNET + DRT   |  0.58     |
| Human          |  0.95     |
| Solver         |  1.00     |


## Methods and Results
### Experiment 1: Standalone Language Model
To look at the components of the SM individually, we first tested the LM on the ground-truth attributes of RPM puzzles: *shapes, colour, size* and *angle*. This experiment is done to assess the capabilities of the Language Model to solve the RPM with perfect information. For the Language Model FlanT5 was chosen, because it is the largest, freely available open-source model, it does not require an API key, and it is trained on chain-of-thought. This means that it can reason about logical problems quite well [[4]](#flant5). Several sizes of the FlanT5 model were used.

As shown in Table 2, the number of parameters in the LM is highly important in its understanding of logical problems, the XL model having nearly double the accuracy of the L model. We would have also liked to use the XXL version, although with its 11B parameters, it did not fit on the LISA cluster. FlanT5's performance drop from XL to L suggests that using even fewer parameters would not yield worthwhile results. Informal testing on the base (220M) and small (60M) versions of FlanT5 showed exactly that, and thus they were not investigated further.


Table 2. Standalone Language Model Test Results:

| Model (num. params) | Precision | Recall | F1   | Accuracy |
|---------------------|-----------|--------|------|---|
| LM: FT5-L (770M)    | 0.46      | 0.45   | 0.44 | 0.45 |
| LM: FT5-XL (3B)     | 0.81      | 0.78   | 0.78 | 0.78 |

The following prompt was used to solve the puzzles:

***You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}.***

In the {}, the attributes of the individual figures (shape, colour, angle, size) are explained.

### Experiment 2: Socratic Model
In this experiment, we used three different architectures, each incorporating a unique visual module. The initial two architectures employed VLMs, namely CLIP and BLIP, while the third one relied on traditional OpenCV methods. For a visual depiction of the pipeline, please consult Figure 3.

![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/25672fcd-722e-4566-aaec-df6f186b705b)

Figure 3. The proposed pipeline starts by taking a puzzle and annotating its attributes using a visual module. These attributes are then combined to form a prompt, which is passed down to the language model. Finally, the language model generates the final answer.

CLIP (Contrastive Language-Image Pretraining) is a state-of-the-art Visual-Linguistic Model (VLM) designed for zero-shot classification [[5]](#CLIP). It achieves this by training on a large dataset of image-text pairs, enabling it to understand and relate images and their corresponding textual descriptions. By leveraging this pretraining, CLIP can generalize to classify images even without specific training on the target classes, making it a powerful tool for zero-shot classification tasks.

<img src="https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/e2c444d8-688d-4657-a534-2b71e46b27db" data-canonical-src="https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/e2c444d8-688d-4657-a534-2b71e46b27db" width="60%" height="60%" />

Figure 4. The OpenCV method explained.
<br><br>

The OpenCV method obtains the shape of the image by extracting the corners of the puzzle shape with edge detection [[6]](#OpenCV). Then the number of vertices can be drawn from the number of corners, which results in a name for a shape. If the amount of vertices is 4, it is a square, etc. The colour is detected by getting the RGB values of the centre of the shape. The size is obtained by comparing the size of the shape with the overall size of the square of the puzzle.


[TODO: FILL IN AND DISCUSS RESULTS OF TABLE 2]
The results of experiment 2 are shown in table 2.

Table 2. Socratic Model Test Results:

| SM (num. params) | Precision | Recall | F1   | Accuracy |
|---------------------|-----------|--------|------|----|
| CLIP + FT5-L (TODO/770M)    | 0.136      | 0.136   | 0.120 | 0.136 |
| CLIP + FT5-XL (TODO/3B)     | 0.127     | 0.118   | 0.097 | 0.118 |
| BLIP + FT5-L (TODO/770M)     | 0.136      | 0.127   | 0.107 | 0.127 |
| BLIP + FT5-XL (TODO/3B)     | TODO      | TODO   | TODO | TODO |
| CV2 + FT5-L (-/770M)     | 0.136      | 0.136   | 0.132 | 0.136 |
| CV2 + FT5-XL (-/3B)     | 0.126      | 0.129   | 0.122 | 0.129 |



### Experiment 3: OpenFlamingo
![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/645e0197-5211-42bd-80d6-cdbbf5b9c7cc)

Figure 5. Schematic representation of OpenFlamingo [[3]](#flam).

We also compare the Socratic Model to Flamingo [[3]](#flam), a multi-modal VLM for few-shot learning. It uses gates to constrain the LM with the encoded vision input, which is the RPM in this case. As Flamingo is not open source, [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) was used instead.

OpenFlamingo, which uses LLama as its Language Model (LM) with 7B parameters and CLIP as a vision encoder, was expected to outperform the Socratic Model due to the higher parameter count in the LM and the similarity of the vision encoders used. Surprisingly, the opposite is true. OpenFlamingo exhibits significant difficulties in comprehending the problem, performing even worse than random guessing (1/8 or 0.125 accuracy).

It is important to note that OpenFlamingo's primary training focus lies on captioning and Visual Question Answering (VQA) tasks, rather than accurate logical reasoning. Moreover, its training set predominantly consists of photographs, lacking exposure to geometric shapes. This limitation becomes evident as OpenFlamingo tends to provide solutions involving only shape numbers 3, 7, and 8, completely disregarding shapes 1, 2, 4, 5, and 6.

However, based on the findings from Experiment 1, it is evident that Flamingo, a unique architecture that combines a Vision encoder with a Large Language Model, has the potential to effectively solve the Raven dataset.

Table 3. OpenFlamingo Test Results:

| Model (num. params) | Precision | Recall | F1   | Accuracy |
|---------------------|-----------|--------|------|---|
| VLM: OpenFlamingo (9B)       | 0.04      | 0.11   | 0.13 | 0.11 |


## Discussion
[TODO: SUMMARISE RESULTS AND INSIGHTS FROM RESEARCH]

Table 4. TODO: Comparison of SM and OpenFlamingo accuracies to baselines.

| Model | Accuracy          |
|---------------|-------------------|
| **Our method** |   |
| CLIP + Flant5 |            |
| OpenFlamingo  |  0.13     |
| **Baseline**              |
| ResNET +DRT   |  0.58     |
| Human         |  0.95     |
| Solver        |  1.00     |


### Future Work
[TODO: DESCRIBE FUTURE WORK (prompt tuning, few-shot tests, different dataset, different models, finetuning)]

## References
<a id="sm"></a> [[1]](https://arxiv.org/abs/2204.00598) Andy Zeng, Adrian Wong, Stefan Welker, Krzysztof Choromanski, Federico Tombari, Aveek Purohit, Michael S. Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, Pete Florence:
**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language.** *arXiv preprint CoRR abs/2204.00598* (2022)

<a id="raven"></a> [[2]](https://arxiv.org/abs/1903.02741) Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu:
**RAVEN: A Dataset for Relational and Analogical Visual rEasoNing.** *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5317-5327)* (2019)

<a id="flam"></a> [[3]](https://arxiv.org/abs/2204.14198) Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karén Simonyan:
**Flamingo: a Visual Language Model for Few-Shot Learning.** *Advances in Neural Information Processing Systems, 35, 23716-23736* (2022)

<a id="flant5"></a> [[4]](https://arxiv.org/abs/2210.11416) Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei:
**Scaling Instruction-Finetuned Language Models** *arXiv preprint arXiv:2210.11416* (2022)

<a id="CLIP"></a> [[5]](https://arxiv.org/abs/2103.00020) Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others:
**Learning transferable visual models from natural language supervision** *arXiv preprint arXiv:2103.00020* (2021)

<a id="OpenCV"></a> [[6]](http://roswiki.autolabor.com.cn/attachments/Events(2f)ICRA2010Tutorial/ICRA_2010_OpenCV_Tutorial.pdf) Bradski, G.:
**The OpenCV library** *Dr. Dobb's Journal of Software Tools* (2022)
