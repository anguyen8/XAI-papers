# Papers on Explainable Artificial Intelligence

This is an on-going attempt to consolidate all interesting efforts in the area of understanding / interpreting / explaining / visualizing machine learning models.

---------------------------------------

# GUI tools
* DeepVis: Deep Visualization Toolbox. _Yosinski et al. 2015_ [code](https://github.com/yosinski/deep-visualization-toolbox) | [pdf](http://yosinski.com/deepvis)

# Demos and code
* https://github.com/tensorflow/lucid

# Surveys
* Methods for Interpreting and Understanding Deep Neural Networks. _Montavon et al. 2017_ [pdf](https://arxiv.org/pdf/1706.07979.pdf)
* The Mythos of Model Interpretability. _Lipton 2016_ [pdf](https://arxiv.org/abs/1606.03490)
* Towards A Rigorous Science of Interpretable Machine Learning _Doshi-Velez & Kim. 2017_ [pdf](https://arxiv.org/pdf/1702.08608.pdf)
* Visualizations of Deep Neural Networks in Computer Vision: A Survey. _Seifert et al. 2017_ [pdf](https://link.springer.com/chapter/10.1007/978-3-319-54024-5_6)
* How convolutional neural network see the world - A survey of convolutional neural network visualization methods. _Qin et al. 2018_ [pdf](https://arxiv.org/abs/1804.11191)
* A brief survey of visualization methods for deep learning models from the perspective of Explainable AI. _Chalkiadakis 2018_ [pdf](https://www.macs.hw.ac.uk/~ic14/IoannisChalkiadakis_RRR.pdf)

# Visualizing Preferred Stimuli
## Activation Maximization
* AM: Visualizing higher-layer features of a deep network. _Erhan et al. 2009_ [pdf](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network)
* DeepVis: Understanding Neural Networks through Deep Visualization. _Yosinski et al. 2015_ [pdf](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf) | [url](http://yosinski.com/deepvis)
* MFV: Multifaceted Feature Visualization: Uncovering the different types of features learned by each neuron in deep neural networks. _Nguyen et al. 2016_ [pdf](http://www.evolvingai.org/files/mfv_icml_workshop_16.pdf) | [code](https://github.com/Evolving-AI-Lab/mfv)
* DGN-AM: Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. _Nguyen et al. 2016_ [pdf](anhnguyen.me/project/synthesizing) | [code](https://github.com/Evolving-AI-Lab/synthesizing)
* PPGN: Plug and Play Generative Networks. _Nguyen et al. 2017_ [pdf](anhnguyen.me/project/ppgn/) | [code](https://github.com/Evolving-AI-Lab/ppgn)
* Feature Visualization. _Olah et al. 2017_ [url](https://distill.pub/2017/feature-visualization)
* Diverse feature visualizations reveal invariances in early layers of deep neural networks. _Cadena et al. 2018_ [pdf](https://arxiv.org/pdf/1807.10589.pdf)

## Real images / Segmentation Masks
* Object Detectors Emerge in Deep Scene CNNs. Zhou et al. 2015 [pdf](https://arxiv.org/abs/1412.6856)
* __Network Dissection__ Quantifying Interpretability of Deep Visual Representations. Bau et al. 2017 [url](http://netdissect.csail.mit.edu/) | [pdf](http://netdissect.csail.mit.edu/final-network-dissection.pdf)
* __Net2Vec__ Quantifying and Explaining how Concepts are Encoded by Filters in Deep Neural Networks. Fong & Vedaldi 2018 [pdf](https://arxiv.org/abs/1801.03454)

# Heatmaps / Attribution
### White-box
* Learning how to explain neural networks: PatternNet and PatternAttribution [pdf](https://arxiv.org/abs/1705.05598)
* A Theoretical Explanation for Perplexing Behaviors of Backpropagation-based Visualizations. _Nie et al. 2018_ [pdf](https://arxiv.org/abs/1805.07039)
* A Taxonomy and Library for Visualizing Learned Features in Convolutional Neural Networks [pdf](https://arxiv.org/pdf/1606.07757.pdf)
* How Important Is a Neuron? _Dhamdhere et al._ 2018 [pdf](https://arxiv.org/pdf/1805.12233.pdf)
* CAM:
* Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. _Selvaraju et al. 2017_ [pdf](https://arxiv.org/abs/1610.02391)
* Grad-CAM++:
* Unreliable saliency maps
* LRP: Beyond saliency: understanding convolutional neural networks from saliency prediction on layer-wise relevance propagation [pdf](https://arxiv.org/abs/1712.08268)
* DTD: Explaining NonLinear Classification Decisions With Deep Tayor Decomposition [pdf](https://arxiv.org/abs/1512.02479)
* Regional Multi-scale Approach for Visually Pleasing Explanations of Deep Neural Networks. _Seo et al. 2018_ [pdf](https://arxiv.org/pdf/1807.11720.pdf)

### Black-box
* RISE: Randomized Input Sampling for Explanation of Black-box Models. _Petsiuk et al. 2018_ [pdf](https://arxiv.org/pdf/1806.07421.pdf)
* LIME: Why should i trust you?: Explaining the predictions of any classifier. _Ribeiro et al. 2016_ [pdf](https://arxiv.org/pdf/1602.04938.pdf) | [blog](https://homes.cs.washington.edu/~marcotcr/blog/lime/)

# Inverting Neural Networks
* Understanding Deep Image Representations by Inverting Them [pdf](https://arxiv.org/abs/1412.0035)
* Inverting Visual Representations with Convolutional Networks [pdf](https://arxiv.org/abs/1506.02753)
* Neural network inversion beyond gradient descent [pdf](http://opt-ml.org/papers/OPT2017_paper_38.pdf)

# Bayesian approaches
* Yang, S. C. H., & Shafto, P. Explainable Artificial Intelligence via Bayesian Teaching. NIPS 2017 [pdf](http://shaftolab.com/assets/papers/yangShafto_NIPS_2017_machine_teaching.pdf)

# Distilling DNNs into more interpretable models
* Interpreting CNNs via Decision Trees [pdf](https://arxiv.org/abs/1802.00121)
* Distilling a Neural Network Into a Soft Decision Tree [pdf](https://arxiv.org/abs/1711.09784)

# Learning to explain
* Deep Learning for Case-Based Reasoning through Prototypes [pdf](https://arxiv.org/pdf/1710.04806.pdf)
* Unsupervised Learning of Neural Networks to Explain Neural Networks [pdf](https://arxiv.org/abs/1805.07468)

# Understanding via Mathematical and Statistical tools
* Understanding Deep Architectures by Interpretable Visual Summaries [pdf](https://arxiv.org/pdf/1801.09103.pdf)
* A Peek Into the Hidden Layers of a Convolutional Neural Network Through a Factorization Lens. _Saini et al. 2018_ [pdf](https://arxiv.org/abs/1806.02012)

# Applications
* Explainable AI for Designers: A Human-Centered Perspective on Mixed-Initiative Co-Creation [pdf](http://www.antoniosliapis.com/papers/explainable_ai_for_designers.pdf)
* ICADx: Interpretable computer aided diagnosis of breast masses. _Kim et al. 2018_ [pdf](https://arxiv.org/abs/1805.08960)

# Others
* Neural Network Interpretation via Fine Grained Textual Summarization. _Guo et al. 2018_ [pdf](https://arxiv.org/pdf/1805.08969.pdf)
