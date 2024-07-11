## Knowledge Graph Triples

### Datastore creation from monolingual data

For KG triples extraction, we use [mREBEL](https://huggingface.co/Babelscape/mrebel-base) model.
To create the datastore, three scripts need to be run:

1. Run the `src/kg-extraction/main.py` script to extract KG triples.
```sh
python src/kg-extraction/main.py -d <training-data-path-file> -o <output-directory-path>
```

2. Run the `src/kg-extraction/process_kbs.py` script to compile all extracted triples in a single csv file.

3. Run the `src/ragmt/generate_datastore.py` script to finally generate the embeddings for each triple, and store it in [FAISS](https://github.com/facebookresearch/faiss) index.
