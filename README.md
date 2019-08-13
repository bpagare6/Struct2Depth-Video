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

