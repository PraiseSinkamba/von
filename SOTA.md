Peer Comment

[–]EstarriolOfTheEast 9 points 17 hours ago* 



To me, this Von project looks better than Bespoke Nimble in that Von's author looks to have thought more carefully about calibration and training data type. Architecturally too, bidirectional attention is much more sample efficient than a causal decoder for NLI classification. Bespoke Nimble's main advantage is model scale and Qwen's extensive knowledge.



What distinguishes Jev is calibration, speed and zeroshot inference for classification but most of these open attempts are overfocused on classification and speed.



So based on technical remit, Von is the best approach I've come across so far. Its main limitation is due to not enough training data. Unfortunately, we are waiting for a large context pretrained purpose built 9B-27B encoder. Until then, the largest modern options are T5GemmaV2 4B Encoder or DiffusionGemma Decoder mode (has bidirectional attention, although more work than T5 would be needed to get Diffusion working as a jev-style NLI classifier).



I'd suggest Von's author continue with ModernBERT and try to beat and build upon the prior art to get something flexible that punches way above its weight class.

===

https://github.com/vinnylarouge/jevlike

https://huggingface.co/collections/MoritzLaurer/zeroshot-classifiers

https://flaviocopes.com/jev/

https://archerhume.com/posts/jevs-architecture-unmasked/?v=3

https://typesafe.ai/blog/introducing-system-one-models-and-jev

https://morethanamachine.com/posts/jev-style-decisions-dgx-spark/
