from typing import Dict
from kedro.pipeline import Pipeline
from dataforseo_project.pipelines.pipeline import create_pipeline
from dataforseo_project.pipelines.pipeline_seed import create_seed_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    seo = create_pipeline()
    seed = create_seed_pipeline()
    return {"__default__": seo, "seo": seo, "seed": seed}
