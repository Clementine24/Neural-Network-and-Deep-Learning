# Orc-BERT Augmentor
This project is based on [Sketch-BERT](https://github.com/avalonstrel/SketchBERT). 

Our Orc-BERT model has similar but smaller network than Sketch-BERT model.

In order to generate new samples from the few-shot training set, function `generate` is added in `main.py`.

Starts from tuning hyper parameters in `models/SketchTransformer/config/sketch_transformer.yml`. 

Then set the path of .npz data files in `generate_dataset.py`, and run it.

Then run:
`python main.py train models/SketchTransformer/config/sketch_transformer.yml`
in shell to train the Orc-BERT model.

Finally, change the files in `models/SketchTransformer/config/sketch_transformer.yml` and run
`python main.py generate models/SketchTransformer/config/sketch_transformer.yml` in shell to generate new training samples.
