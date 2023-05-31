# Solving Visual-Language Logic Puzzles with Socratic Models
28/05/2023 | Sergei Agaronian, Theodoor Akkerboom, Maarten Drinhuyzen, Wenkai Pan, Marius Strampel

## Introduction
Zeng et al. (2022) [[1]](#sm) introduce Socratic
Models (SMs). SMs enable the composition of multiple large, pre-trained models in a modular framework, allowing them to exchange information with one another and to tackle new multimodal challenges without the need for fine-tuning. These models are based on large pretrained models and can vary their capabilities wildly depending on the domain they are trained on. By combining such models, SMs have been shown to be competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval. Moreover, SMs can additionally be extended using non-neural modules, such as a web-crawler. As such, SMs open up a range of new applications including answering free-form questions about the egocentric video, engaging in multimodal assistive dialogue with people by interfacing with external APIs and databases, and robot perception and planning.

To see how far the adaptability of SMs goes, several experiments were run to test SMs capability to solve logical vision-language problems out of the box. In our experiments, SM is compared to an oracle LM and Flamingo models as baselines.

## Socratic Models
### Background
The core concept of SMs is to harness what Zeng et al. (2022) call differences in common sense knowledge. Visual Language Models (VLMs), for example, are trained on image captions, and thus develop understanding of the connection between images and language. Language Models (LMs) are trained on additional corpora that do not need to possess a visual aspect, such as novels or recipes. As such, they are applicable to a wide variety of linguistic tasks. By using SMs, we can harness these complementary differences in knowledge and apply them to both existing and new tasks out-of-the-box.

![image](https://gyazo.com/8d2f8d1a893ed836f6c9dc12ef927753.png)

Figure from [[1]](#sm): Differing and complementary commonsense knowledge learned by different foundation models.

[TODO: ADD MORE INFORMATION]

[TODO: ADD FOLLOWUP RESEARCH TO SHOW STATE OF SM RESEARCH]

### Strengths and Weaknesses
The idea behind Socratic Models is to chain existing models as is, without any training or fine-tuning. This eliminates the need for additional compute resources and time-consuming training processes. By leveraging pre-trained models, Socratic Models can quickly provide intelligent and context-aware responses. This approach also ensures that the system remains up-to-date with the latest information and developments, as it operates based on existing state-of-the-art. In addition, as the state of the art advances, modules can be quickly swapped. The absence of training requirements allows for effortless integration and implementation of Socratic Models across various applications and platforms, making it a versatile and efficient solution for natural language understanding and generation tasks.

Additionally, SM's modular design allows chaining of models with different modalities, such as VLMs with LMs. This allows SMs to have multimodal understanding, greatly increasing the range of problems SMs can tackle. Research by Zeng et al. (2022) demonstrates that SMs can thus be competitive with state-of-the-art on image captioning, video-to-text retrieval, and more.

It is worth noting that VLMs, trained using contrastive learning, may not possess the same level of language understanding as LMs. By chaining LMs with VLMs, the overall model's language comprehension is enhanced, resulting in improvements over using VLMs alone.

However, chaining multiple models in Socratic Models introduces the possibility of error propagation through its constituent components. If the initial language model provides an incorrect or biassed response, subsequent models in the chain may continue to build upon that flawed information, leading to a cascade of inaccurate or misleading answers. Addressing error propagation requires careful monitoring, evaluation, and periodic intervention to correct any erroneous responses and maintain the overall quality and reliability of the system.

Another potential challenge lies in the lack of a shared embedding space among the foundation models used in Socratic Models. Without a common embedding space, seamless transfer of knowledge between models becomes difficult, and language acts as the medium for their interaction. This can pose a problem if the language understanding of a model is limited, either in general or domain-specific, as information might get lost in translation. Without a shared representation, it becomes more difficult to leverage the strengths of individual models and to ensure consistent understanding and coherent responses across the chain. This lack of a common embedding space may result in inconsistencies, conflicting information, and difficulties in maintaining a cohesive and coherent conversational flow. All of this is on top of language itself often being ambiguous.

## Contribution
This research project intends to explore the potential of an SM pipeline, incorporating a VLM and an LM, to solve logical puzzles represented as images, such as Raven Progressive Matrices which are explained in detail below. The experiments will focus on assessing the general reasoning capabilities of the language model in the proposed pipeline and compare the results to baselines, the language model by itself and Flamingo. The research objectives are to assess the performance of a Socratic model on logical puzzles, compare it to existing methods, and explore its potential to extend to other applications. [TODO: IS PREV SENTENCE CORRECT? MOST OF THIS IS NO LONGER PART OF OUR RESEARCH RIGHT?]. The significance of the proposed work is that it will provide insight into the SMs logic understanding and its capabilities to use said understanding to solve logical puzzles. It will also provide a comparison between existing models and the proposed one.

The proposed work consists of constructing a pipeline using the Socratic model framework and using an open-source implementation of different VLMs and LMs, such as Clip, Open-Assistant, Codex and FlanT5. Also traditional Computer Vision methods will be tested to extract the shapes of the Raven Progressive Matrices. The exact models are determined by testing the performance of different models. The performance of this pipeline will be assessed both quantitatively and qualitatively. Quantitative results will be compared to the baselines.

### Raven Progressive Matrices
The Socratic Model is tested on the Center Single subset of the Raven Progressive Matrices dataset by Zhang et al. (2019) [[2]](#raven) in all experiments. An RPM consists of a three by three grid of items, with the final item missing. There are eight potential solutions, of which the model must pick the missing shape to finish the pattern. An example is shown in figure 2 below. This study aims to provide a proof-of-concept for solving logical puzzles such as RPMs with SMs and thus sticks to the centre single subset to avoid logistic difficulties. Each item contains a single shape (triangle, square, circle, hexagon or pentagon). The correct answer is identified by the shape itself and its additional attributes, namely size, colour and angle.<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Additional subsets exist, namely with multiple shapes and even overlapping shapes in each of the sections/answers and further research could tackle those, along with entirely different datasets.

<br>![image](https://gyazo.com/c6d0bcea47b740917a436c0f8ab3411c.png)<br> Figure 2. RPM Examples. Top: RPM problem with missing answer in the bottom-right corner, bottom: possible answers for the model to choose from.


### Baselines
The baseline we use to compare our Socratic Model to, is from the paper [[2]](#raven) of the Socratic Model. They used a Dynamically Residual Tree, to solve the Raven Dataset. The Dynamically Residual Tree is a Tree traversal algorithm, which solves the puzzle by going through the nodes. To get the Visual Encoding of the shapes, they used ResNET. They also compared it to human performance and a solver.

Table 1. Baseline for comparison

| Model| accuracy   |
|------|-------------|
| ResNET +DRT   |  0.58     |
| Human         |  0.95     |
| Solver        |  1.00     |


## Methods and Results
### Experiment 1
To look at the components of the SM individually, we first tested the LM on the groundtruth attributes of the shapes of the RPM; shapes,color, size and angle. This experiment is done to assess the capibilities of the Language Model to solve the RPM with perfect information. For the Language Model, we choose FlanT5, because it is openaccess, doesn't require an API key and is trained on chain-of-thought. This means that it can, according to [[4]](#flant5), reason about mathematical problems.  Several sizes of the FlanT5 model were used. As shown in table 2, the number of parameters in the LM matters greatly in its understanding of logical problems, with the XL model having nearly double the accuracy of the L model. We would have liked to use the XXL version also, although with its 11B parameters, it did not fit on the LISA cluster. Additionally, we did not use smaller FlanT5 models that are available, as the performance drop from XL to L suggests going smaller still would not be fruitful. 

Table 2. Experiment 1

| Model (num. params) | Precision | Recall | F1   |
|---------------------|-----------|--------|------|
| LM: FT5-L (770M)    | 0.46      | 0.45   | 0.45 |
| LM: FT5-XL (3B)     | 0.81      | 0.78   | 0.78 |

As a prompt we used the following format; **You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}.**

In the {}, the attributes of the shapes are explained.

### Experiment 2 The Socratic Model
![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/25672fcd-722e-4566-aaec-df6f186b705b)<br> Figure 3. SM pipeline using CLiP and BLiP as VLMs.

With the Socratic Model we tested 2 VLMs; CLIP and BLIP and a traditional Computer Visions algorithm from OpenCV. These methods are used to get the visual understanding of the image, which is passed to the LLM for solving the puzzle. 

<img src="https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/e2c444d8-688d-4657-a534-2b71e46b27db" data-canonical-src="https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/e2c444d8-688d-4657-a534-2b71e46b27db" width="60%" height="60%" />
<br> Figure 4. The OpenCV method explained. [[5]]("OpenCV")
<br><br>

The OpenCV method obtains the shape of the image by extracting the corners of the puzzle shape with Edge Detection[[5]](#OpenCV). Then the amount of vertices can be invered from the amount of corners, which results in a name for a shape. If the amount of vertices is 4, it is a square etc. The color is detected by getting the RGB values of the center of the shape. The size is obtained by comparing the size of the shape to the overall size of the square of the puzzle.


[TODO: FILL IN AND DISCUSS RESULTS OF TABLE 2]

| SM (num. params) | Precision | Recall | F1   |
|---------------------|-----------|--------|------|
| CLiP + FT5-L (TODO/770M)    | TODO      | TODO   | TODO |
| CLiP + FT5-XL (TODO/3B)     | TODO      | TODO   | TODO |
| BLiP + FT5-L (TODO/770M)     | TODO      | TODO   | TODO |
| BLiP + FT5-XL (TODO/3B)     | TODO      | TODO   | TODO |
| CV2 + FT5-L (-/770M)     | TODO      | TODO   | TODO |
| CV2 + FT5-XL (-/3B)     | TODO      | TODO   | TODO |

Table 2. Results for different SM configurations.

### Experiment 3 OpenFlamingo
![image](https://github.com/TAkkerboom/DL2_socratic_models_team_2/assets/131353365/645e0197-5211-42bd-80d6-cdbbf5b9c7cc)<br> Figure 5. Schematic representation of OpenFlamingo [[3]](#flam). 

We also compare the Socratic Model to Flamingo [[3]](#flam), a multi-modal VLM for few-shot learning. It uses gates to constrain the LM with the encoded vision input, which is the RPM in this case. As flamingo is not opensource, [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) was used instead. OpenFlamingo uses LLama as LM with 7B parameters and CLIP as a vision encoder. Because the vision encoder is equal to the vision encoder we use in the Socratic Model in Experiment 2, and the amount of parameters is higher for the LLM, the performance should in theory be better than the Socratic Model. The opposite is true.  We see that the OpenFlamingo model struggles to understand the the problem, having lower accuracy than random guessing (1/8 or 0.125). Although OpenFlamingo uses a 7B parameter Language Model, which is more parameters than FlanT5-XL, OpenFlamingo is not trained to handle logical puzzles accurately, with both a vision and a cognition aspect. OpenFlamingo is trained mostly on captioning and VQA, not on logical reasoning. The trainingset of OpenFlamingo also mostly consists of photographs, not on geometric shapes. This could be seen by the fact that, within the multiple choices, OpenFlamingo only gives the shape number 3,7 and 8 as solution. Shapes 1,2,4,5 and 6 are never given as solution. Because of the high accuracy of the different Language Models given perfect information, shown in Experiment 1, Flamingo has potential to solve the Raven dataset, because it fuses an Vision encoder and a Large Language Model in one architecture, thereby preventing error accumulation.

Table 3. OpenFlamingo

| Model (num. params) | Precision | Recall | F1   |
|---------------------|-----------|--------|------|
| VLM: OpenFlamingo (9B)       | 0.04      | 0.11   | 0.13 |

## Conclusion
| Model | Accuracy          |
|---------------|-------------------|
| **Our method** |   |
| CLIP + Flant5 |            |
| OpenFlamingo  |  0.13     |
| **Baseline**              |
| ResNET +DRT   |  0.58     |
| Human         |  0.95     |
| Solver        |  1.00     |


## References
<a id="sm"></a> [[1]](https://arxiv.org/abs/2204.00598) Andy Zeng, Adrian Wong, Stefan Welker, Krzysztof Choromanski, Federico Tombari, Aveek Purohit, Michael S. Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, Pete Florence:
**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language.** *arXiv preprint CoRR abs/2204.00598* (2022)

<a id="raven"></a> [[2]](https://arxiv.org/abs/1903.02741) Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu:
**RAVEN: A Dataset for Relational and Analogical Visual rEasoNing.** *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5317-5327)* (2019)

<a id="flam"></a> [[3]](https://arxiv.org/abs/2204.14198) Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karén Simonyan:
**Flamingo: a Visual Language Model for Few-Shot Learning.** *Advances in Neural Information Processing Systems, 35, 23716-23736* (2022)

<a id="flant5"></a> [[4]](https://arxiv.org/abs/2210.11416) Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei:
**Scaling Instruction-Finetuned Language Models** *arXiv preprint arXiv:2210.11416* (2022)

<a id="OpenCV"></a> [[5]](http://roswiki.autolabor.com.cn/attachments/Events(2f)ICRA2010Tutorial/ICRA_2010_OpenCV_Tutorial.pdf) Bradski, G.:
**The OpenCV library** *Dr. Dobb's Journal of Software Tools* (2022)
