# Canonical References

> The authoritative list for this project. Full title + authors + venue/year + link, with
> a one-line relevance note. Verification: `arXiv-ID` = metadata fetched from the arXiv
> API; `manual` = classic/software, confirm DOI on first cite; `unverified` = not yet
> confirmed. BibTeX: `references.bib` (keep in sync).
>
> **Citation rule:** always use the full method/paper name and attribution in prose —
> e.g. "Fourier Domain Adaptation (FDA) [Yang & Soatto, CVPR 2020]".

---

## A. Datasets and benchmarks

### A1. ACDC: The Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception
- **Authors:** Christos Sakaridis, Haoran Wang, Ke Li, René Zurbrügg, Arpit Jadon, Wim Abbeloos, Daniel Olmeda Reino, Luc Van Gool, Dengxin Dai
- **Venue/Year:** ICCV 2021 · **arXiv:2104.13395** · Verification: `arXiv-ID`
- **Relevance:** our **target domain**; fog/rain/night/snow with per-condition GT; official val (406) is our scored benchmark. (Note: the arXiv title/variant uses "Robust Semantic Driving Scene Perception"; the ICCV'21 title uses "…Semantic Driving Scene Understanding". Confirm the proceedings title at submission.)

### A2. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
- **Authors:** Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, Trevor Darrell
- **Venue/Year:** CVPR 2020 · **arXiv:1805.04687** · Verification: `arXiv-ID`
- **Relevance:** our **source domain** (clear/daytime subset); supplies the large clear-weather detection pool.

### A3. The Cityscapes Dataset for Semantic Urban Scene Understanding
- **Authors:** Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele
- **Venue/Year:** CVPR 2016 · **arXiv:1604.01685** · Verification: `arXiv-ID`
- **Relevance:** the standard driving benchmark and the origin of most DA/foggy-detection protocols; useful related-work anchor.

### A4. Semantic Foggy Scene Understanding with Synthetic Data
- **Authors:** Christos Sakaridis, Dengxin Dai, Luc Van Gool
- **Venue/Year:** IJCV 2018 · **arXiv:1708.07819** · Verification: `arXiv-ID`
- **Relevance:** the canonical **synthetic-fog** protocol (Foggy Cityscapes); the archetype of our S3/S5 synthetic-to-real design and the discipline of a physics-based fog model.

### A5. DAWN: Vehicle Detection in Adverse Weather Nature Dataset
- **Authors:** Mourad A. Kenk, Mahmoud Hassaballah
- **Venue/Year:** preprint 2020 · **arXiv:2008.05402** · Verification: `arXiv-ID`
- **Relevance:** small real adverse-weather detection dataset; cite as evidence that real adverse-weather detection benchmarks are scarce (motivates ACDC usage).

### A6. Canadian Adverse Driving Conditions Dataset
- **Authors:** Matthew Pitropov, Danson Garcia, Jason Rebello, Michael Smart, Carlos Wang, Krzysztof Czarnecki, Steven Waslander
- **Venue/Year:** IJRR 2021 (confirm) · **arXiv:2001.10117** · Verification: `arXiv-ID`
- **Relevance:** adverse-weather (snow) driving dataset; related work on real adverse data.

### A7. How Hard Is Snow? A Paired Domain Adaptation Dataset for Clear and Snowy Weather: CADC+
- **Authors:** Mei Qi Tang, Sean Sedwards, Chengjie Huang, Krzysztof Czarnecki
- **Venue/Year:** preprint 2025 · **arXiv:2506.16531** · Verification: `arXiv-ID`
- **Relevance:** recent paired clear↔snow benchmark; useful comparison point and evidence that paired adverse data is still being built.

### A8. Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather
- **Authors:** Mario Bijelic, Tobias Gruber, Fahim Mannan, Florian Kraus, Werner Ritter, Klaus Dietmayer, Felix Heide
- **Venue/Year:** CVPR 2020 · **arXiv:1902.08913** · Verification: `arXiv-ID`
- **Relevance:** fog perception benchmark + multimodal fusion; a camera-only contrast to our setting.

### A9. NightOwls: A Pedestrians at Night Dataset
- **Authors:** Lukas Neumann, Michelle Karg, Shanshan Zhang, Christian Scharfenberger, Eric Piegert, Sarah Mistr, Olga Prokofyeva, Robert Thiel, Andrea Vedaldi, Andrew Zisserman, Bernt Schiele
- **Venue/Year:** 2019 (dataset) · Verification: `manual` (not found on arXiv)
- **Relevance:** night-time pedestrian detection benchmark; support for "night is a distinct low-illumination problem".

---

## B. Detection architectures and backbones

### B1. You Only Look Once: Unified, Real-Time Object Detection (YOLOv1)
- **Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi · **CVPR 2016** · **arXiv:1506.02640** · `arXiv-ID`
- **Relevance:** origin of the one-stage family we use.

### B2. YOLOv4: Optimal Speed and Accuracy of Object Detection
- **Authors:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao · **arXiv 2020** · **arXiv:2004.10934** · `arXiv-ID`
- **Relevance:** introduces **Mosaic** augmentation + bag-of-freebies; source of the default augmentation our S1 uses.

### B3. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors
- **Authors:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao · **CVPR 2023** · **arXiv:2207.02696** · `arXiv-ID`
- **Relevance:** one-stage anchor-free design; optional generalization check.

### B4. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
- **Authors:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao · **ICLR 2024** · **arXiv:2402.13616** · `arXiv-ID`
- **Relevance:** optional generalization check.

### B5. Ultralytics YOLOv8 / YOLO11
- **Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu (Ultralytics) · Software 2023–2024 · `manual`
- **Relevance:** the detector family we train (locked to YOLOv8n); cite the software docs and pin the version (`8.4.147`).

### B6. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- **Authors:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun · **NeurIPS 2015** · **arXiv:1506.01497** · `arXiv-ID`
- **Relevance:** two-stage baseline used throughout DA-detection literature.

### B7. Feature Pyramid Networks for Object Detection
- **Authors:** Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie · **CVPR 2017** · **arXiv:1612.03144** · `arXiv-ID`
- **Relevance:** multi-scale backbone feature used by most detectors.

### B8. Focal Loss for Dense Object Detection (RetinaNet)
- **Authors:** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár · **ICCV 2017** · **arXiv:1708.02002** · `arXiv-ID`
- **Relevance:** one-stage loss/architecture context.

### B9. End-to-End Object Detection with Transformers (DETR)
- **Authors:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko · **ECCV 2020** · **arXiv:2005.12872** · `arXiv-ID`
- **Relevance:** transformer-based alternative; related-work breadth.

### B10. Microsoft COCO: Common Objects in Context
- **Authors:** Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár · **ECCV 2014** · **arXiv:1405.0312** · `arXiv-ID`
- **Relevance:** pretraining + the mAP metric definition we report (mAP@50, mAP@50-95).

---

## C. Domain adaptation for object detection

### C1. Domain Adaptive Faster R-CNN for Object Detection in the Wild
- **Authors:** Yuhua Chen, Wen Li, Christos Sakaridis, Dengxin Dai, Luc Van Gool · **CVPR 2018** · **arXiv:1803.03243** · `arXiv-ID`
- **Relevance:** the founding adversarial DAOD method; the baseline framing most DAOD work inherits.

### C2. Strong-Weak Distribution Alignment for Adaptive Object Detection (SW-DA)
- **Authors:** Kuniaki Saito, Yoshitaka Ushiku, Tatsuya Harada, Kate Saenko · **CVPR 2019** · **arXiv:1812.04798** · `arXiv-ID`
- **Relevance:** domain-classifier alignment; standard comparison.

### C3. Unbiased Mean Teacher for Cross-domain Object Detection (UMT)
- **Authors:** Jinhong Deng, Wen Li, Yuhua Chen, Lixin Duan · **CVPR 2021** · **arXiv:2003.00707** · `arXiv-ID`
- **Relevance:** teacher-student self-training for DAOD; relevant if we add a semi-supervised stage.

### C4. SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection
- **Authors:** Wuyang Li, Xinyu Liu, Yixuan Yuan · **CVPR 2022** · **arXiv:2203.06398** · `arXiv-ID`
- **Relevance:** strong modern DAOD; shows the complexity bar a simple augmentation method must justify itself against.

### C5. Prior-based Domain Adaptive Object Detection for Hazy and Rainy Conditions
- **Authors:** Vishwanath A. Sindagi, Poojan Oza, Rajeev Yasarla, Vishal M. Patel · **ECCV 2020** · **arXiv:1912.00070** · `arXiv-ID`
- **Relevance:** **directly related** — uses physical priors (atmospheric scattering / rain model) for adverse-weather DAOD; a key positioning target for S3/S5 physics-based synthesis.

### C6. DA-RAW: Domain Adaptive Object Detection for Real-World Adverse Weather Conditions
- **Authors:** Minsik Jeon, Junwon Seo, Jihong Min · **ICRA 2024** · **arXiv:2309.08152** · `arXiv-ID`
- **Relevance:** **closest protocol match** — separates weather/style gap for real adverse weather; a comparison point and a source of protocol conventions.

### C7. Domain Adaptive Object Detection for Autonomous Driving under Foggy Weather
- **Authors:** Jinlong Li, Runsheng Xu, Jin Ma, Qin Zou, Jiaqi Ma, Hongkai Yu · **WACV 2023** (confirm) · **arXiv:2210.15176** · `arXiv-ID`
- **Relevance:** fog-specific DAOD; comparison for the fog condition.

### C8. AWADA: Attention-Weighted Adversarial Domain Adaptation for Object Detection
- **Authors:** Maximilian Menke, Thomas Wenzel, Andreas Schwung · **WACV 2023** (confirm) · **arXiv:2208.14662** · `arXiv-ID`
- **Relevance:** adversarial DAOD with attention weighting; the "AWADA" tag. (Note: the project notes said "AWADA (2022)"; this is the verified paper matching the tag — confirm you meant this one.)

### C9. An Unsupervised Domain Adaptive Approach for Multimodal 2D Object Detection in Adverse Weather Conditions
- **Authors:** George Eskandar, Robert A. Marsden, Pavithran Pandiyan, Mario Döbler, Karim Guirguis, Bin Yang · **IEEE IV 2022** (confirm) · **arXiv:2203.03568** · `arXiv-ID`
- **Relevance:** adverse-weather UDA for 2D detection; multimodal, but a protocol reference.

### C10. Seeking Similarities over Differences: Similarity-based Domain Alignment for Adaptive Object Detection
- **Authors:** Farzaneh Rezaeianaran, Rakshith Shetty, Rahaf Aljundi, Daniel Olmeda Reino, Shanshan Zhang, Bernt Schiele · **ICCV 2021** · **arXiv:2110.01428** · `arXiv-ID`
- **Relevance:** verified ICCV'21 alignment method; kept partly because the old "ViSGA (ICCV 2021)" note could not be verified (see `09`).

### C11. Multiscale Domain Adaptive YOLO for Cross-Domain Object Detection
- **Authors:** Mazin Hnewa, Hayder Radha · preprint 2021 (confirm venue) · **arXiv:2106.01483** · `arXiv-ID`
- **Relevance:** DA for YOLO specifically; relevant to our YOLOv8 setting.

### C12. Integrated Multiscale Domain Adaptive YOLO
- **Authors:** Mazin Hnewa, Hayder Radha · preprint 2022 (confirm venue) · **arXiv:2202.03527** · `arXiv-ID`
- **Relevance:** likely the Hnewa CVPR-Work/2023 line; a candidate to replace the unverified "MIC" tag (see `09`).

### C13. Domain-Adversarial Training of Neural Networks (DANN)
- **Authors:** Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky · **JMLR 2016** · **arXiv:1505.07818** · `arXiv-ID`
- **Relevance:** foundational adversarial DA.

---

## D. Frequency-domain methods (S4 core)

### D1. FDA: Fourier Domain Adaptation for Semantic Segmentation
- **Authors:** Yanchao Yang, Stefano Soatto · **CVPR 2020** · **arXiv:2004.05498** · `arXiv-ID`
- **Relevance:** **the S4 method** — swap low-frequency amplitude between source and target, keep phase. Our prior B3 run used this and did not beat S1; the knowledge hub records its exact formulation and known limitations.

### D2. A Fourier-based Framework for Domain Generalization (FACT)
- **Authors:** Qinwei Xu, Ruipeng Zhang, Ya Zhang, Yanfeng Wang, Qi Tian · **CVPR 2021** · **arXiv:2105.11120** · `arXiv-ID`
- **Relevance:** amplifies amplitude perturbation + mixes amplitude/phase across domains for **DG**; a stronger Fourier baseline and the basis for phase/amplitude arguments.

### D3. PAGen: Phase-guided Amplitude Generation for Domain-adaptive Object Detection
- **Authors:** Shuchen Du, Shuo Lei, Feiran Li, Jiacheng Li, Daisuke Iso · preprint 2025 · **arXiv:2511.22029** · `arXiv-ID`
- **Relevance:** **closest method to the original SM-WCFA idea** (phase-guided amplitude generation for DAOD). Must be positioned against explicitly; our S-study is a comparative benchmark, not a new Fourier method.

### D4. The Importance of Phase in Signals
- **Authors:** Alan V. Oppenheim, Jae S. Lim · **Proc. IEEE 1981** · `manual`
- **Relevance:** the classic justification that **phase carries structure** — why geometry-preserving amplitude mixing keeps boxes valid (used in S4/S3 reasoning).

---

## E. Weather synthesis, rendering, and image translation

### E1. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)
- **Authors:** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros · **ICCV 2017** · **arXiv:1703.10593** · `arXiv-ID`
- **Relevance:** the standard unpaired clear↔weather translation backbone; a possible S5 baseline or related work.

### E2. Multimodal Unsupervised Image-to-Image Translation (MUNIT)
- **Authors:** Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz · **ECCV 2018** · **arXiv:1804.04732** · `arXiv-ID`
- **Relevance:** multimodal translation (diverse weather styles); relevant if S5 is learned rather than physics-based.

### E3. Contrastive Learning for Unpaired Image-to-Image Translation (CUT)
- **Authors:** Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu · **ECCV 2020** · **arXiv:2007.15651** · `arXiv-ID`
- **Relevance:** modern unpaired translation; lighter alternative to CycleGAN.

### E4. Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions (IA-YOLO)
- **Authors:** Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, Lei Zhang · **AAAI 2022** · **arXiv:2112.08088** · `arXiv-ID`
- **Relevance:** **detection-specific adverse-weather adaptation** (differentiable image processing front-end); a strong comparison point and a contrast to training-time-only synthesis.

### E5. Photorealistic Rendering of Rain Streaks
- **Authors:** Kshitiz Garg, Shree K. Nayar · **ACM SIGGRAPH/TOG 2006** · `manual`
- **Relevance:** physics of rain-streak appearance (motion blur, illumination); basis for a defensible S3/S5 rain model.

### E6. Vision and Rain
- **Authors:** Kshitiz Garg, Shree K. Nayar · **IJCV 2007** · `manual`
- **Relevance:** imaging model of rain; the reference for "rain is a dynamic, depth-dependent phenomenon", motivating why naive streaks are a weak S3.

### E7. Koschmieder's Law (Theorie der horizontalen Sichtweite)
- **Authors:** Harald Koschmieder · **1924** · `manual`
- **Relevance:** the atmospheric-scattering/visibility model `I = J·t + A(1−t)` used for physics-based fog in S3/S5.

### E8. Single Image Haze Removal Using Dark Channel Prior
- **Authors:** Kaiming He, Jian Sun, Xiaoou Tang · **CVPR 2009 / TPAMI 2011** · `manual`
- **Relevance:** dark-channel prior gives a **depth/transmission proxy** for calibrating fog synthesis (S5) without a depth network.

---

## F. Image restoration and low-level vision (context for S2/S5)

### F1. EnlightenGAN: Deep Light Enhancement without Paired Supervision
- **Authors:** Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, Zhangyang Wang · **IEEE TPAMI 2021** · **arXiv:1906.06972** · `arXiv-ID`
- **Relevance:** unpaired low-light enhancement; night-condition related work.

### F2. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE)
- **Authors:** Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong · **CVPR 2020** · **arXiv:2001.06826** · `arXiv-ID`
- **Relevance:** reference-free night enhancement; context for why night is an illumination problem (S2/S5).

### F3. Restormer: Efficient Transformer for High-Resolution Image Restoration
- **Authors:** Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang · **CVPR 2022** · **arXiv:2111.09881** · `arXiv-ID`
- **Relevance:** strong restoration backbone; a possible restoration-vs-detection comparison.

### F4. Multi-Stage Progressive Image Restoration (MPRNet)
- **Authors:** Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao · **CVPR 2021** · **arXiv:2102.02808** · `arXiv-ID`
- **Relevance:** deraining/dehazing restoration; context.

### F5. FFA-Net: Feature Fusion Attention Network for Single Image Dehazing
- **Authors:** Xu Qin, Zhilin Wang, Yuanchao Bai, Xiaodong Xie, Huizhu Jia · **AAAI 2020** · **arXiv:1911.07559** · `arXiv-ID`
- **Relevance:** dehazing baseline; restored-then-detect comparison.

---

## G. Data augmentation

### G1. AutoAugment: Learning Augmentation Policies from Data
- **Authors:** Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le · **CVPR 2019** · **arXiv:1805.09501** · `arXiv-ID`
- **Relevance:** learned augmentation policy; background for condition-aware policies (S6).

### G2. RandAugment: Practical Automated Data Augmentation with a Reduced Search Space
- **Authors:** Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le · **CVPRW 2020** · **arXiv:1909.13719** · `arXiv-ID`
- **Relevance:** the `auto_augment: randaugment` default in our S1; must be acknowledged when defining S2 as "S1 + photometric" (S1 already contains photometric ops).

### G3. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
- **Authors:** Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan · **ICLR 2020** · **arXiv:1912.02781** · `arXiv-ID`
- **Relevance:** robustness-oriented mixing augmentation; conceptual neighbour of weather mixing.

### G4. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation
- **Authors:** Samuel G. Müller, Frank Hutter · **CVPR 2021** · **arXiv:2103.10158** · `arXiv-ID`
- **Relevance:** strong augmentation with no search; a sanity comparison for S2.

### G5. Domain Generalization with MixStyle
- **Authors:** Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang · **ICLR 2021** · **arXiv:2104.02008** · `arXiv-ID`
- **Relevance:** feature-statistic mixing; relevant to condition-aware feature-level ideas and to the retired WSM direction.

### G6. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN)
- **Authors:** Xun Huang, Serge Belongie · **ICCV 2017** · **arXiv:1703.06868** · `arXiv-ID`
- **Relevance:** image statistic (mean/std) transfer; the principled basis for illumination/colour calibration (S2/S5) and for arguing FDA's tint limit.

---

## H. Domain generalization, robustness, test-time adaptation

### H1. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- **Authors:** Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel · **IROS 2017** · **arXiv:1703.06907** · `arXiv-ID`
- **Relevance:** the sim-to-real framing of S3/S5 (synthetic weather → real ACDC).

### H2. Tent: Fully Test-Time Adaptation by Entropy Minimization
- **Authors:** Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell · **ICLR 2021** · **arXiv:2006.10726** · `arXiv-ID`
- **Relevance:** test-time adaptation option; labelled future work (not in S0–S6).

### H3. Mean Teachers Are Better Role Models: Weight-averaged Consistency Targets Improve Semi-supervised Deep Learning Results
- **Authors:** Antti Tarvainen, Harri Valpola · **NeurIPS 2017** · **arXiv:1703.01780** · `arXiv-ID`
- **Relevance:** teacher-student consistency; background for UMT/self-training stages.

### H4. A Simple Semi-Supervised Learning Framework for Object Detection (STAC)
- **Authors:** Kihyuk Sohn, Zizhao Zhang, Chun-Liang Li, Han Zhang, Chen-Yu Lee, Tomas Pfister · preprint 2020 · **arXiv:2005.04757** · `arXiv-ID`
- **Relevance:** semi-supervised detection; future work for using unlabeled ACDC.

---

## I. Adverse-weather detection and comparative studies

### I1. D-YOLO: A Robust Framework for Object Detection in Adverse Weather Conditions
- **Authors:** Zihan Chu · preprint 2024 · **arXiv:2403.09233** · `arXiv-ID`
- **Relevance:** recent YOLO-based adverse-weather detector; related work.

### I2. Object Detection Under Rainy Conditions for Autonomous Vehicles: A Review of State-of-the-Art and Emerging Techniques
- **Authors:** Mazin Hnewa, Hayder Radha · **IEEE T-ITS 2021** (confirm) · **arXiv:2006.16471** · `arXiv-ID`
- **Relevance:** a survey of rain-specific detection; useful to justify per-condition analysis.

### I3. Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions
- **Authors:** Fox Pettersen, Hong Zhu · preprint 2026 · **arXiv:2602.12902** · `arXiv-ID`
- **Relevance:** recent robustness/benchmark study; direct related comparison for our per-weather evaluation.

### I4. From Filters to VLMs: Benchmarking Defogging Methods through Object Detection and Segmentation Performance
- **Authors:** Ardalan Aryashad, Parsa Razmara, Amin Mahjoub, Seyedarmin Azizi, Mahdi Salmani, Arad Firouzkouhi · preprint 2025 · **arXiv:2510.03906** · `arXiv-ID`
- **Relevance:** benchmark that scores restoration methods by downstream detection — methodological neighbour of our "does synthesis/adaptation help detection" question.

### I5. Bridging Clear and Adverse Driving Conditions
- **Authors:** Yoel Shapiro, Yahia Showgan, Koustav Mullick · preprint 2025 · **arXiv:2508.13592** · `arXiv-ID`
- **Relevance:** recent clear↔adverse bridging study; check for overlap before claiming novelty.

### I6. Nighttime Pedestrian Detection Based on Fore-Background Contrast Learning
- **Authors:** He Yao, Yongjun Zhang, Huachun Jian, Li Zhang, Ruzhong Cheng · preprint 2024 · **arXiv:2408.03030** · `arXiv-ID`
- **Relevance:** night-specific detection; supports the "night is illumination-dominated" argument.

### I7. Towards Robust 3D Object Detection in Rainy Conditions
- **Authors:** Aldi Piroli, Vinzenz Dallabetta, Johannes Kopp, Marc Walessa, Daniel Meissner, Klaus Dietmayer · **IEEE IV 2023** (confirm) · **arXiv:2310.00944** · `arXiv-ID`
- **Relevance:** rain robustness for 3D detection; adjacent evidence.

---

## J. Unverified tags from earlier notes (do not cite yet)

- **"MIC (CVPR 2023)"** — could not verify a paper titled "Multi-Intensity-Aware Consistency…". Likely the Hnewa & Radha line (`C11`/`C12`). Confirm the intended paper before citing.
- **"ViSGA (ICCV 2021)"** — could not verify a paper with this name. The verified ICCV'21 similarity-alignment paper is `C10`. Confirm or drop.
- **"PAGen (2025)"** — now verified (`D3`).
