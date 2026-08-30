# Research on Chinese Character and Kanji Recognition

The most efficient systems for handwritten Kanji and related Han-character recognition combine visual modelling with the script's internal structure: strokes, radicals, components, glyph variants, and reading context. Recent work has expanded the task beyond isolated-character classification to zero-shot recognition, cross-temporal historical perception, line-level handwriting recognition, document restoration, scene-text understanding, and efficient edge deployment.

Much of the relevant literature is published as **Chinese character recognition (CCR)** because the methods operate on Han characters shared across Chinese and Japanese writing. Such work is technically relevant to Kanji, but Chinese benchmark results should not be described as direct Japanese results unless the study evaluates Japanese data explicitly.

## 1. Methods Focusing on Structure

Several techniques improve recognition by decomposing characters into reusable units. This is particularly useful for large inventories, rare characters, writer variation, and zero-shot recognition.

1. **Radical-level Ideograph Encoding:** This approach uses embeddings of radicals composing Chinese and Japanese ideographs rather than relying only on independent character embeddings (Radical-level Ideograph Encoder for RNN-based Sentiment Analysis of Chinese and Japanese, Ke & Hagiwara, 2017).
   - The radical-level strategy reduces the effective vocabulary and shares information between characters with common components.
   - It is suitable for Chinese and Japanese text systems with limited parameters.
   - The original approach combines a CNN word-feature encoder with a bidirectional RNN document-feature encoder.

2. **Hierarchical Decomposition and Nearest-Neighbor Classification:** A framework for Japanese historical characters, _kuzushiji_, learns reusable character parts and transfers knowledge from synthesized fonts to historical handwriting (Japanese historical character recognition by focusing on character parts, Ishikawa, Miyazaki, & Omachi, 2024).
   - Shared components address the severe sample imbalance found in historical Japanese documents.
   - The method supports few-shot and zero-shot recognition by matching learned part representations.
   - It reports nearly 48% accuracy for zero-sampled _kuzushiji_, where naive classification methods were unable to recognize the characters.

3. **Radical-based Online Recognition Systems:** Online recognizers can combine radical appearance, geometric relations, vector quantization, Markov random fields, and structured dictionaries. These methods are useful when digital ink or pen trajectories are available rather than only raster images (Advances in online handwritten recognition in the last decades, Ghosh, Sen, Obaidullah, et al., 2022, citing Ma & Liu, 2009; Zhu & Nakagawa).

4. **Hierarchical Grammatical Modelling:** Stochastic Context-Free Grammars combined with Hidden Markov Models have been proposed to model character generation hierarchically and support writer-independent online recognition (Advances in online handwritten recognition in the last decades, Ghosh, Sen, Obaidullah, et al., 2022, citing Ota, Yamamoto, Sako, & Sagayama, 2007).

## 2. Lightweight and Hierarchical Deep Learning

Large character inventories make a conventional flat softmax expensive. Hierarchical encodings, prototype matching, quantization, and small backbones reduce classification cost and make deployment on mobile or edge hardware more realistic.

### HierCode

**[HierCode: A Lightweight Hierarchical Codebook for Zero-shot Chinese Text Recognition](https://arxiv.org/abs/2403.13761v1)** (March 2024, Zhang et al.) proposes a multi-hot hierarchical codebook rather than a conventional one-hot classification layer.

- A binary-tree representation encodes character relationships and supports similarity-based inference.
- Prototype learning gives characters distinctive code representations.
- A lightweight backbone such as [MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html) can reduce the classification burden.
- The method is designed for out-of-vocabulary recognition when a new character shares structural information with known characters.
- The paper evaluates handwritten, scene, document, historical, and web-text recognition settings.

### Hi-GITA

**[Zero-Shot Chinese Character Recognition with Hierarchical Multi-Granularity Image-Text Aligning](https://arxiv.org/abs/2505.24837v1)** (May 2025, Zhu et al.) was already present in the original document and is retained here as an existing method rather than a new addition.

- Hi-GITA aligns visual and textual character representations at stroke, radical, and complete-character levels.
- Its image encoder extracts hierarchical representations from character images, while its text encoder represents stroke sequences, radical sequences, and character descriptions.
- Multi-Granularity Fusion Modules connect the image and text streams.
- A Fine-Grained Decoupled Image-Text Contrastive loss aligns corresponding representations across levels.
- The authors report roughly 20% improvement in handwritten-character and radical zero-shot settings over earlier methods.
- The method is relevant to Kanji because Chinese characters and Japanese Kanji share compositional stroke and component structure, although the reported experiments are Chinese-character experiments.

The original document also contains more specific Hi-GITA figures, including exact accuracy ranges, parameter counts, latency, loss weights, and patch counts. Those figures should be retained only after checking the full paper tables and implementation; the abstract establishes the architecture and the approximately 20% zero-shot improvement, but not all of those detailed values.

### GL-HPN: Global-Local Hierarchical Perception Network

**[Zero-Shot Chinese Character Recognition via Global-Local Dual-Branch Alignment and Hierarchical Inference](https://arxiv.org/abs/2605.08814)** (May 2026, Cao, Xu, & Diao) proposes a Global-Local Hierarchical Perception Network for zero-shot Chinese character recognition.

- The global branch learns whole-character image and IDS representations for efficient coarse retrieval.
- The local branch uses patch-token interaction to distinguish component-level differences that a single global vector may miss.
- A structure-filtering mask suppresses IDS operators that are structurally meaningful but do not correspond directly to visual entities.
- A coarse-to-fine inference strategy retrieves candidates globally and applies local re-ranking only to the Top- candidates.
- Parameter-free multiplicative fusion combines normalized global and local posterior scores.
- The paper reports competitive performance across zero-shot splits, especially in low-resource settings, while substantially reducing large-scale candidate-retrieval cost.

GL-HPN is a natural successor to HierCode and Hi-GITA in the document's zero-shot section. Its key contribution is not simply another structural encoder, but a scalable inference policy that reserves expensive local matching for a small candidate set.

## 3. Stroke, Radical, and Sequential Recognition

RNNs remain relevant when the input is a sequence of pen coordinates, stroke images, or characters in a line. Transformers are increasingly useful when larger datasets and more compute are available.

1. **Stroke-based RNN:** Processes each stroke with coordinate, timing, direction, and pen-state information.
2. **Radical-sequence RNN:** Represents a character as an ordered sequence of components and processes the sequence with a bidirectional RNN.
3. **CNN-RNN hybrid:** Uses CNN layers for raster appearance and RNN layers for stroke or line-level sequence information.
4. **Attention-enhanced sequence models:** Focus processing on discriminative strokes or components.
5. **Stroke-level self-supervision:** Learns from pen trajectories or stroke images and can support recognition when character labels are scarce.

### DTRNet: Dual Text-Radical Decoding

**[DTRNet: Dual Text-Radical Decoding for Handwritten Chinese Text Recognition with Faked Character Detection](https://arxiv.org/abs/2608.05848)** (August 2026, Li, Zhu, & Huang) introduces a line-level framework for recognizing handwritten Chinese text while detecting deliberately fabricated or structurally invalid characters.

- The text branch performs context-aware line-level transcription.
- The radical branch predicts legal Ideographic Description Sequences (IDS) as independent structural evidence.
- A character is flagged as potentially fake when the transcribed character and its predicted radical structure do not agree with the lexicon.
- IDS-Guided Confidence Adjustment (IGCA) refines recognition predictions using structural evidence during inference.
- The architecture preserves character-wise structural verification without giving up the efficiency of line-level recognition.
- The paper reports strong recognition performance alongside interpretable radical-level evidence and states that code, checkpoints, and processed data are publicly available.
- The work was accepted as an oral paper at ACM Multimedia 2026.

DTRNet is relevant to Kanji recognition in two ways. First, its dual-decoder design offers a practical way to combine context-aware text recognition with component-level verification. Second, the idea of a structurally invalid or suspicious character can be adapted to quality control, handwriting tutoring, OCR uncertainty detection, and historical-document transcription.

## 4. CNN and Vision-Transformer Recognition

An ensemble of three CNNs achieved 96.43% classification accuracy on the top 150 classes of the imbalanced Kuzushiji-Kanji dataset (Recognition of Handwritten Japanese Characters Using Ensemble of Convolutional Neural Networks, Solis, Zarkovacki, Ly, & Atyabi, 2023).

- CNN ensembles can deliver high accuracy on restricted or frequent-character inventories.
- Their disadvantages are larger deployment cost, slower inference, and limited support for unseen classes compared with structurally encoded methods.
- Transfer learning reduced training time in one component on the K-49 dataset.
- Vision Transformers are useful when character images, local patches, and document context must be integrated, but they usually require more data and compute than compact CNN baselines.

## 5. Historical and Degraded Japanese Documents

Clean character benchmarks do not represent archival conditions. Fading, stains, bleed-through, seals, warped pages, and unusual layouts can dominate recognition error.

### DKDS

**[DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization](https://arxiv.org/abs/2511.09117)** (2025, Ju et al.; revised versions released in 2026) introduces a benchmark for degraded _kuzushiji_ documents containing seal interference.

- The benchmark separates character-and-seal detection from document binarization.
- Detection baselines use recent YOLO models to locate _kuzushiji_ characters and seals.
- Binarization experiments compare traditional methods, K-means-assisted approaches, GAN-based methods, and an improved conditional GAN.
- DKDS demonstrates that recognition quality depends on document restoration and object detection before classification.
- The [DKDS project page and code repository](https://github.com/RuiyangJu/DKDS) provide dataset and implementation resources.

### Restoration-Guided Kuzushiji Recognition

**[Restoration-Guided Kuzushiji Character Recognition Framework under Seal Interference](https://arxiv.org/abs/2602.19086)** (February 2026, Ju, Yamashita, Kameko, & Mori) develops a detection-restoration-recognition pipeline for documents in which seals overlap characters.

- Stage 1 detects characters and seals.
- Stage 2 restores or removes seal interference.
- Stage 3 classifies the restored character with a Vision Transformer-based _kuzushiji_ classifier.
- The paper reports 98.0% precision and 93.3% recall for YOLOv12-medium on its detection test set.
- Restoration improved Top-1 classification accuracy from 93.45% to 95.33% in an ablation study.
- The main implication is architectural: historical OCR should be treated as a detection-restoration-recognition pipeline, not merely as a classifier applied to a clean crop.

## 6. Cross-Temporal Historical Character Recognition

### Chronicles-OCR

**[Chronicles-OCR: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters](https://arxiv.org/abs/2605.11960)** (May 2026, Li et al.) introduces a benchmark for evaluating vision-language models across the historical evolution of Chinese writing.

- It covers the Seven Chinese Scripts and is designed to measure visual perception across major morphological and topological shifts.
- The benchmark contains 2,800 strictly balanced images from media ranging from tortoise shells to paper-based calligraphy.
- Four tasks are defined: cross-period character spotting, fine-grained archaic-character recognition through visual referring, ancient-text parsing, and script classification.
- A Stage-Adaptive Annotation Paradigm accommodates the substantial differences between historical writing stages.
- The benchmark is designed to separate visual perception from higher-level semantic reasoning.
- Its relevance to Kanji lies in evaluating transfer across historical glyph forms, variant shapes, and writing traditions rather than assuming that one modern glyph representation is sufficient.

Chronicles-OCR should be distinguished from ordinary OCR datasets. It is not merely a larger list of character labels; it tests whether a model can recognize visual forms whose structure changes over time. This makes it particularly useful for research on historical Kanji, variant forms, epigraphy, and comparative Sino-Japanese script history.

## 7. Japanese Scene Text and Vision-Language Benchmarks

### JaWildText

**[JaWildText: A Benchmark for Vision-Language Models on Japanese Scene Text Understanding](https://arxiv.org/abs/2603.27942)** (March 2026, Maeda & Okazaki) addresses Japanese text in real-world images rather than only scanned documents.

- The benchmark contains 3,241 instances from 2,961 images captured in Japan.
- It covers 3,643 unique character types and approximately 1.12 million annotated characters.
- It includes dense scene-text visual question answering, receipt key-information extraction, and handwriting OCR.
- It tests mixed scripts, vertical writing, layout variation, and large Japanese character inventories.
- Evaluation of open-weight VLMs shows that Japanese recognition remains a major bottleneck, particularly for Kanji.
- The [JaWildText project](https://arxiv.org/abs/2603.27942) is useful for diagnosing Japanese-script failure modes, not merely ranking models.

## 8. Large-Scale Datasets

### MegaHan97K

**[MegaHan97K: A Large-Scale Dataset for Mega-Category Chinese Character Recognition with over 97K Categories](https://arxiv.org/abs/2506.04807)** (June 2025, Zhang et al.) introduces a benchmark with 97,455 Chinese-character categories.

- It supports the GB18030-2022 character standard and substantially exceeds the scale of earlier public character datasets.
- The dataset contains handwritten, historical, and synthetic subsets.
- Its design targets long-tail recognition, where rare characters have far fewer real examples than common characters.
- Benchmarking identifies storage demands, morphologically similar characters, and zero-shot recognition as major challenges.
- The [official MegaHan97K repository](https://github.com/SCUT-DLVCLab/MegaHan97K) provides dataset and benchmark information.
- For Kanji research, MegaHan97K is best treated as a transferable Han-character benchmark rather than a replacement for Japanese-specific evaluation.

### JieZi and Ancient Character Exegesis

**[JieZi: A Large-Scale Expert-Audited Dataset and Benchmark for Ancient Chinese Character Exegesis](https://arxiv.org/abs/2608.11741)** (August 2026, Li, He, Cao, Liu, Cheng, & Jin) extends ancient-character research beyond recognition into structured scholarly interpretation.

- The paper formulates **Ancient Chinese Character Exegesis (ACCE)** as a vision-language question-answering task.
- ACCE has four progressive levels: basic character identification, glyph-form analysis, meaning exegesis, and diachronic evolution analysis.
- JieZi-Dataset contains more than 500,000 question-answer pairs constructed with expert-designed templates, source-text references, and human verification.
- JieZi-Bench contains held-out reference answers curated from authoritative lexicographic works.
- Experiments show that multimodal models perform comparatively well on basic identification but struggle with glyph analysis, semantic reasoning, and diachronic understanding.
- Fine-tuning on JieZi-Dataset improves performance across all four levels.
- Code and data are available through the [JieZi project link](https://arxiv.org/abs/2608.11741).

JieZi is not a conventional Kanji classifier. Its importance is that it provides a model for connecting recognition with philological explanation. For historical Kanji research, analogous tasks could include identifying a glyph, describing its structural form, linking it to a Japanese variant, explaining historical meaning, and tracing changes across Chinese and Japanese usage.

### Stroke-Level Handwriting Data

**[A Stroke-Level Large-Scale Database of Chinese Character Handwriting](https://arxiv.org/abs/2509.05335)** (September 2025, Xu et al.) provides stroke-level handwriting data and tools for trajectory research.

- The database was collected from 42 writers, each writing 1,200 characters in a handwriting-to-dictation task.
- Stroke-level data supports research on online handwriting, stroke segmentation, writer variation, and trajectory-based recognition.
- It complements bitmap datasets because it preserves temporal writing structure.
- The resource is relevant to Kanji systems that need to model pen movement rather than only final glyph appearance.

### MCCD

**[MCCD: A Multi-Attribute Chinese Calligraphy Character Dataset](https://arxiv.org/abs/2507.06948)** (2025) associates character images with style, period, and calligrapher attributes.

- It supports recognition across multiple script styles and historical contexts.
- The attributes enable experiments in writer identification, style transfer, calligraphic recognition, and character evolution.
- Its relevance to Japanese material is indirect, but the evaluation principles apply to Kanji written in different historical and calligraphic styles.

## 9. Historical Text and General OCR Models

### CHURRO

**[CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition](https://arxiv.org/abs/2509.19768)** (September 2025, Semnani et al.) presents a 3-billion-parameter open-weight VLM specialized for historical text recognition.

- CHURRO-DS combines 155 historical corpora, 99,491 pages, 22 centuries of textual heritage, and 46 language clusters.
- The model reports 82.3% normalized Levenshtein similarity on printed text and 70.1% on handwritten text in its evaluation.
- Its significance for Kanji is primarily document-level: historical recognition requires adaptation to script variation, irregular layouts, degradation, and context.
- CHURRO should not be interpreted as a Japanese-Kanji-specific model unless Japanese results are reported separately.

### PaddleOCR-VL-1.5

**[PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing](https://arxiv.org/abs/2601.21957)** (January 2026, revised April 2026) is a compact VLM-oriented document parser.

- It reports 94.5% on OmniDocBench v1.5 in the paper's evaluation.
- It adds seal recognition and text spotting to document parsing.
- Its Real5-OmniDocBench evaluation targets scanning, skew, warping, screen photography, and illumination changes.
- The model is relevant to Kanji document systems as a general document-level baseline, but specialized Japanese and historical evaluation remains necessary.

### PP-OCRv6

**[PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks](https://arxiv.org/abs/2606.13108)** (June 2026) presents OCR models for server, mobile, and edge deployment.

- The architecture uses a unified MetaFormer-style block with structural reparameterization.
- Separate model tiers target different deployment budgets.
- The paper reports an in-house recognition accuracy of 83.2% and detection Hmean of 86.2% for its medium model.
- The tiny model is reported as 3.9 times faster than PP-OCRv5_mobile on Intel Xeon CPU.
- These figures are in-house results, not direct Kanji benchmark results.
- The work reinforces the value of specialized, quantized, and edge-oriented OCR rather than assuming that a large VLM is always preferable.

## 10. Comparison of Current Approaches

| Method or resource     | Primary task                                  | Structural information                           | Rare-character value                        | Japanese relevance                                    | Deployment profile  |
| ---------------------- | --------------------------------------------- | ------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------- | ------------------- |
| Radical-level encoding | Text and character modelling                  | Radicals/components                              | Medium                                      | Directly motivated by Chinese and Japanese ideographs | Lightweight         |
| HierCode               | Large-vocabulary recognition                  | Hierarchical codebook                            | High                                        | Transferable, but Chinese benchmark                   | Edge-friendly       |
| Hi-GITA                | Zero-shot character recognition               | Stroke, radical, character, image-text alignment | High                                        | Transferable structural method                        | Small-to-medium     |
| GL-HPN                 | Zero-shot retrieval                           | Global/local image and IDS branches              | High                                        | Transferable to Kanji inventories                     | Retrieval-efficient |
| DTRNet                 | Line recognition and fake-character detection | Text plus legal IDS/radical decoding             | High for structural validation              | Transferable to handwriting quality control           | Line-level          |
| CNN ensemble           | Isolated handwritten recognition              | Mostly visual                                    | Low for unseen classes                      | Direct Japanese _kuzushiji_ evidence                  | Larger/slower       |
| RG-KCR                 | Degraded _kuzushiji_ recognition              | Detection, restoration, classification           | Not primarily zero-shot                     | Direct Japanese historical evidence                   | Multi-stage         |
| Chronicles-OCR         | Cross-temporal historical perception          | Scripts, glyph evolution, visual referring       | High for historical variants                | Indirect but highly relevant                          | Benchmark           |
| JaWildText             | Japanese scene-text evaluation                | Mixed scripts, layout, context                   | Diagnostic                                  | Direct Japanese evidence                              | Benchmark           |
| MegaHan97K             | Mega-category benchmarking                    | Character categories and subsets                 | Explicit zero-shot and long-tail evaluation | Transferable Han-character evidence                   | Dataset-heavy       |
| JieZi                  | Ancient-character exegesis                    | Glyph, meaning, diachrony                        | High for expert-audited historical analysis | Transferable research framework                       | VLM/dataset         |
| CHURRO                 | Historical document OCR                       | Document context and layout                      | Useful for rare historical text             | Indirect unless Japanese subset used                  | 3B-parameter VLM    |
| PP-OCRv6               | General OCR                                   | Visual and document-level features               | Not primarily zero-shot                     | General multilingual relevance                        | Edge/server tiers   |

## 11. Recommendations for Tsujimoto

For a project targeting approximately 3,000 Kanji classes, evaluation should include Top-1 accuracy, macro accuracy, memory, latency, and performance on rare or unseen classes.

1. **Retain CNN and RNN baselines.** They provide controls for measuring the contribution of structural and multimodal representations.
2. **Retain HierCode as the compact hierarchical baseline.** Its codebook formulation is valuable when the classification layer dominates model size.
3. **Retain Hi-GITA as an existing implementation direction.** Do not list it as a newly discovered method; instead, verify its detailed numerical claims against the full paper.
4. **Add GL-HPN-style global/local retrieval.** Its coarse-to-fine inference is a promising way to scale zero-shot candidate matching.
5. **Add DTRNet-style structural verification.** A second decoder predicting IDS or radical structure could flag uncertain, malformed, or suspicious character predictions.
6. **Add a degradation track.** Use synthetic blur, stains, contrast changes, seal-like overlays, and warped backgrounds, then evaluate preprocessing and recognition separately.
7. **Add a historical-evolution track.** Chronicles-OCR suggests testing visual recognition across historical forms rather than evaluating only modern glyphs.
8. **Add expert-audited explanation tasks.** JieZi provides a model for evaluating identification, glyph analysis, meaning, and diachronic explanation separately.
9. **Use Japanese-specific evaluation.** JaWildText and _kuzushiji_ benchmarks expose failure modes that Chinese-character datasets cannot fully represent.
10. **Use macro accuracy and per-class recall.** Large datasets can hide failure on rare characters behind high micro accuracy.
11. **Benchmark edge deployment explicitly.** Record model size, RAM, CPU latency, energy use, and quantized accuracy.
12. **Separate transfer claims from direct evidence.** Chinese-character results support architectural hypotheses for Kanji, but Japanese evaluation is required before claiming Kanji superiority.

## 12. Updated Research Priorities

The strongest current directions are:

- **Hierarchical multimodal representation:** Aligning strokes, radicals, IDS, glyphs, and holistic character images.
- **Coarse-to-fine inference:** Retrieving a small candidate set globally and applying expensive local comparison only to that set.
- **Dual-path structural validation:** Combining line-level transcription with independent radical or IDS verification.
- **Long-tail and mega-category evaluation:** Testing rare and unseen characters instead of only common classes.
- **Cross-temporal perception:** Measuring robustness across historical scripts, media, glyph variants, and morphological change.
- **Restoration before recognition:** Treating seals, degradation, binarization, and layout as part of the recognition system.
- **Japanese in-the-wild evaluation:** Measuring mixed scripts, vertical writing, handwriting, and real-world capture conditions.
- **Expert-audited multimodal interpretation:** Separating basic recognition from glyph analysis, semantic interpretation, and historical explanation.
- **Efficient specialized OCR:** Comparing small OCR systems with VLMs under realistic edge constraints.
- **Stroke-aware learning:** Preserving digital-ink trajectories and writing order whenever online input is available.

The overall direction is not simply “larger model equals better recognition.” A practical Kanji system will likely combine a compact visual encoder, structural candidate retrieval, Japanese language or dictionary context, uncertainty estimation, and a restoration/detection front end for historical or noisy documents.

## 13. Links and Recent Papers

### Core recognition methods

- [HierCode: A Lightweight Hierarchical Codebook for Zero-shot Chinese Text Recognition](https://arxiv.org/abs/2403.13761v1)
- [Zero-Shot Chinese Character Recognition with Hierarchical Multi-Granularity Image-Text Aligning (Hi-GITA)](https://arxiv.org/abs/2505.24837v1)
- [Zero-Shot Chinese Character Recognition via Global-Local Dual-Branch Alignment and Hierarchical Inference (GL-HPN)](https://arxiv.org/abs/2605.08814)
- [DTRNet: Dual Text-Radical Decoding for Handwritten Chinese Text Recognition with Faked Character Detection](https://arxiv.org/abs/2608.05848)
- [Recognition of Handwritten Japanese Characters Using Ensemble of Convolutional Neural Networks](https://arxiv.org/abs/2306.16688)

### Datasets and historical recognition

- [MegaHan97K](https://arxiv.org/abs/2506.04807)
- [MegaHan97K GitHub repository](https://github.com/SCUT-DLVCLab/MegaHan97K)
- [MCCD](https://arxiv.org/abs/2507.06948)
- [A Stroke-Level Large-Scale Database of Chinese Character Handwriting](https://arxiv.org/abs/2509.05335)
- [DKDS](https://arxiv.org/abs/2511.09117)
- [DKDS GitHub repository](https://github.com/RuiyangJu/DKDS)
- [Restoration-Guided Kuzushiji Character Recognition Framework under Seal Interference](https://arxiv.org/abs/2602.19086)
- [Chronicles-OCR](https://arxiv.org/abs/2605.11960)
- [JieZi](https://arxiv.org/abs/2608.11741)
- [JaWildText](https://arxiv.org/abs/2603.27942)

### Historical and general OCR systems

- [CHURRO](https://arxiv.org/abs/2509.19768)
- [PaddleOCR-VL-1.5](https://arxiv.org/abs/2601.21957)
- [PP-OCRv6](https://arxiv.org/abs/2606.13108)
- [Hashigo: A Next Generation Sketch Interactive System for Japanese Kanji](https://arxiv.org/abs/2504.13940)

### Existing resources retained from the original document

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Lipi Toolkit](http://lipitk.sourceforge.net/hpl-datasets.htm)
- [UNIPEN](http://www.unipen.org/index.html)
- [Jieba](https://github.com/fxsjy/jieba)
- [igraph](https://python.igraph.org/)
- [Mathpix Digital Ink](https://mathpix.com/digital-ink)
- [ML Kit Text Recognition](https://developers.google.com/ml-kit/vision/text-recognition)
- [MyScript Nebo](https://www.nebo.app/)
- [Goodnotes](https://www.goodnotes.com/)
- [MetaMoJi](http://noteanytime.com/en/)

## 14. Scope and Evidence Notes

This document combines Japanese-specific studies with broader Chinese-character recognition research. Chinese-character methods are relevant because Kanji share Han-character structure, but transferability should be experimentally validated. Reported metrics are dataset-specific: accuracy, macro accuracy, normalized edit similarity, precision, recall, and OCR Hmean should not be compared as if they measured the same task.

The 2026 papers cited here are recent preprints or technical reports, although DTRNet and JieZi state acceptance at ACM Multimedia 2026. Claims should still be checked against final proceedings, released code, dataset versions, and benchmark protocols before production decisions are made.
