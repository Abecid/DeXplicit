# Licensing and data sources

The repository’s source code is Apache-2.0 except where third-party notices specify otherwise. This does not relicense external model assets or datasets.

- [HOT3D](https://github.com/facebookresearch/hot3d): sequence data uses CC BY-SA; hand annotations use CC BY-NC-SA; object models have a modified CC BY-SA condition prohibiting sale/incorporation in a product for sale. See the [dataset agreement](https://huggingface.co/datasets/bop-benchmark/hot3d/blob/main/hot3d_dataset_license_agreement.pdf), updated 29 May 2024.
- [OakInk2 official download](https://huggingface.co/datasets/kelvin34501/OakInk-v2), linked from [the project](https://oakink.net/v2/), labels the dataset CC BY-SA 4.0.
- [PhysTwin](https://jianghanxiao.github.io/phystwin-web/): obtain the data and check its distributed terms separately from the code license.
- [Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B): the frozen upstream model is fetched separately under its published terms.
- [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B): obtain backbone/VAE files directly from the upstream repository and retain its terms.
- DINOv3 and MANO: obtain any required assets separately under their respective licenses. Neither set of weights/assets is redistributed here.

## DeXplicit weights

The **DeXplicit 3D model** and **DeXplicit renderer** weights, including the optional compact renderer, are licensed under **Apache-2.0**. See [LICENSE-WEIGHTS](../LICENSE-WEIGHTS).

Apache-2.0 is also used for [Wan-Move's weights](https://huggingface.co/Ruihang/Wan-Move-14B-480P), [Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B), and [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B). Upstream assets and datasets retain their own terms; this license covers the DeXplicit weights distributed here.


No raw datasets or dataset-derived inference fixtures are redistributed in v0.1.0. Users prepare licensed inputs with the documented schema.
