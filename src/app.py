import gradio as gr
import pandas as pd
from huggingface_hub import HfApi

api = HfApi()

# Mapping user-friendly labels to API values
LIBRARY_OPTIONS = ["All", "Gradio", "Streamlit", "Flask", "FastAPI", "Transformers", "Diffusers"]
LICENSE_OPTIONS = ["All", "Apache-2.0", "MIT", "BSD-3-Clause", "GPL-3.0"]
HARDWARE_OPTIONS = ["All", "CPU", "GPU"]
VISIBILITY_OPTIONS = ["All", "Public", "Private"]

SORT_OPTIONS = {
    "Last Modified": SpaceSort.LAST_MODIFIED,
    "First Indexed": SpaceSort.FIRST_INDEXED,
    "Likes": SpaceSort.LIKES,
    "Runs": SpaceSort.RUNS
}
DIRECTION_OPTIONS = ["Descending", "Ascending"]


def search_spaces(query, library, license, tags, visibility, hardware, sort_by, direction, limit):
    # Prepare filters
    lib_filter = None if library == "All" else getattr(SpaceLibraries, library.upper(), None)
    license_filter = None if license == "All" else license
    hardware_filter = None
    if hardware == "CPU":
        hardware_filter = SpaceHardware.CPU
    elif hardware == "GPU":
        hardware_filter = SpaceHardware.GPU
    vis_filter = None
    if visibility == "Public":
        vis_filter = False  # private=False
    elif visibility == "Private":
        vis_filter = True
    # Tags: comma-separated
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    sort_value = SORT_OPTIONS.get(sort_by, SpaceSort.LAST_MODIFIED)
    reverse = False if direction == "Descending" else True

    spaces = api.list_spaces(
        search=query or None,
        library_filter=lib_filter,
        license=license_filter,
        hardware=hardware_filter,
        private=vis_filter,
        sort=sort_value,
        direction="asc" if reverse else "desc",
        limit=limit,
        task=tag_list  # filters by tags/tasks
    )

    # Build list of dicts
    data = []
    for s in spaces:
        data.append({
            "Name": s.id,
            "Author": s.author,
            "Library": s.spaceType,
            "SDK": s.sdk,
            "Tags": ", ".join(s.tags) if hasattr(s, 'tags') else "",
            "Hardware": s.hardware if hasattr(s, 'hardware') else "",
            "Visibility": "Private" if s.private else "Public",
            "Likes": s.likes,
            "Runs": s.runs,
            "Last Modified": s.lastModified,
            "URL": f"https://huggingface.co/spaces/{s.id}"
        })
    return data

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 HF Spaces Explorer with Advanced Filters")
    with gr.Row():
        query = gr.Textbox(label="Search Query", placeholder="Enter keywords...")
        library = gr.Dropdown(LIBRARY_OPTIONS, label="Library", value="All")
        license = gr.Dropdown(LICENSE_OPTIONS, label="License", value="All")
    with gr.Row():
        tags = gr.Textbox(label="Tags (comma-separated)", placeholder="e.g. text-generation, image-classification")
        visibility = gr.Dropdown(VISIBILITY_OPTIONS, label="Visibility", value="All")
        hardware = gr.Dropdown(HARDWARE_OPTIONS, label="Hardware", value="All")
    with gr.Row():
        sort_by = gr.Dropdown(list(SORT_OPTIONS.keys()), label="Sort By", value="Last Modified")
        direction = gr.Radio(DIRECTION_OPTIONS, label="Direction", value="Descending")
        limit = gr.Slider(1, 100, label="Max Results", value=20, step=1)
    search_btn = gr.Button("🔎 Search")
    results = gr.Dataframe(headers=["Name", "Author", "Library", "SDK", "Tags", "Hardware", "Visibility", "Likes", "Runs", "Last Modified", "URL"], label="Results")

    search_btn.click(
        fn=search_spaces,
        inputs=[query, library, license, tags, visibility, hardware, sort_by, direction, limit],
        outputs=results
    )

    gr.Markdown("---")
    gr.Markdown("Enhanced with tag, hardware & visibility filters. Built on `huggingface_hub` and Gradio Blocks.")

if __name__ == "__main__":
    demo.launch(share=True)
