import json
import os
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


def download_model(model_size: str, model_dir: Path):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def set_in_nested_dict(d, keys, val):
    if not keys:
        return val

    d[keys[0]] = set_in_nested_dict(d.get(keys[0], {}), keys[1:], val)

    return d


def save_array(model_dir: Path, subpath: str, array: np.ndarray):
    path = model_dir / "exploded_model" / subpath
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main(model_size: str, model_dir: str):
    model_dir = Path(model_dir) / model_size
    model_dir.mkdir(parents=True, exist_ok=True)

    download_model(model_size, model_dir)

    with open(model_dir / "hparams.json") as hparams_file:
        hparams = json.load(hparams_file)

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name.removeprefix("model/")
        save_array(model_dir, name, array)

        if name.startswith("h"):
            blockname, rest = name.split("/", 1)
            block_idx = int(blockname.removeprefix("h"))
            set_in_nested_dict(params["blocks"][block_idx], rest.split("/"), name)

        else:
            set_in_nested_dict(params, name.split("/"), name)

    with open(model_dir / "model.json", "w") as model_file:
        json.dump(params, model_file)
        print("dumped ")


if __name__ == "__main__":
    import fire

    fire.Fire(main)