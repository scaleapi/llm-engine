# Contributing to LLM Engine

## Updating LLM Engine Documentation

LLM Engine leverages [mkdocs](https://www.mkdocs.org/) to create beautiful, community-oriented documentation.

### Step 1: Clone the Repository

Clone/Fork the [LLM Engine](https://github.com/scaleapi/llm-engine) Repository. Our documentation lives in the `docs` folder.

### Step 2: Install the Dependencies

Dependencies are located in `requirements-docs.txt`, go ahead and pip install those with 

```bash
pip install -r requirements-docs.txt
```

### Step 3: Run Locally

To run the documentation service locally, execute the following command.

```
mkdocs serve
```

This should kick off a locally running instance on [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

As you edit the content in the `docs` folder, the site will be automatically reloaded on each file save.

### Step 4: Editing Navigation and Settings

If you are less familair with `mkdocs`, in addition to the markdown content in the `docs` folder, there is a top-level `mkdocs.yml` file as well that defines the navigation pane and other website settings. If you don't see your page where you think it should be, double-check the .yml file.

### Step 5: Building and Deploying

CircleCI (via `.circleci/config.yml`) handles the building and deployment of our documentation service for us.
