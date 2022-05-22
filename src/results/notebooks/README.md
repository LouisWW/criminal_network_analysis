## Notebook results

---

### Setup the notebooks as slides

    # Activate virtual env
    pipenv shell
    # or
    conda activate criminal_env

    # Add python env to jupyter kernel list
    python -m ipykernel install --user --name=env_name

    # Activate jupyter notebook
    juypter notebook

    # Generate slides and make sure code is removed in the
    # slides
    jupyter nbconvert NOTEBOOK.ipynb --to slides --no-prompt --TagRemovePreprocessor.remove_input_tags={\"to_remove\"} --post serve --SlidesExporter.reveal_theme=simple
