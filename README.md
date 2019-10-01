# struct2depth

## Steps to run the code

1. Download the pre-trained model from [here](https://drive.google.com/file/d/1mjb4ioDRH8ViGbui52stSUDwhkGrDXy8/view)
2. Extract model zip file in struct2depth/trained-models folder
3. Run the code:<br/>
  ```shell
  python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --output_dir output \
    --model_ckpt "./trained-models/model-199160"
  ```
4. Press `ESC` to stop the execution.

### The original code can be found [here](https://github.com/tensorflow/models/tree/master/research/struct2depth)
The respective research paper can be found **V. Casser, S. Pirk, R. Mahjourian, A. Angelova, Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos, AAAI Conference on Artificial Intelligence, 2019**[ https://arxiv.org/pdf/1811.06152.pdf]( https://arxiv.org/pdf/1811.06152.pdf)
