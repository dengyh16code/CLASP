[text](CLASP/README.md)# An Official Code Implementation of CLASP: General-purpose Clothes Manipulation with Semantic Keypoints

If you find this work useful, please cite:
```
@inproceedings{clasp,
  title={General-purpose Clothes Manipulation with Semantic Keypoints},
  author={Yuhong Deng and David Hsu},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2025},
}
  ```

## Installation

**Our code relies on OWL-ViT, SAM, and SD-DINO, which are currently deployed on our internal servers. To use these models locally, please follow the installation instructions provided below.**

1) Installing [[OWL-ViT]](https://huggingface.co/docs/transformers/en/model_doc/owlv2) or another object detector of your choice.

2) Installing [[SAM]](https://github.com/facebookresearch/segment-anything) or another segmentation model of your choice.

3) Installing [[SD+DINO]](https://github.com/Junyi42/geoaware-sc).

4) After installing these models, please replace the code blocks in the original code that were used to call them from our internal servers. The source code of services on our internal servers can be found on `cloud_services/services`.


## Semantic keypoint extraction
**We provide a test code which can be used for semantic keypoint extraction:**


1) Specify your OpenAi api key
```
export OPENAI_API_KEY="your_api_key_here"
```

2) To test the semantic keypoint extraction on your own image, you can replace `test.png` and then run
```
python semantic_keypoints/semantic_keypoint.py
```

3) The semantic keypoints extraction results will be visualized.



