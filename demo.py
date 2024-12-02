import gradio as gr
import ollama
import time

from config import OLLAMA_MODEL_CONFIG

model_name = OLLAMA_MODEL_CONFIG["model_name"]
print(f"[/] USING OLLAMA {model_name} FOR INFERENCE")

def summarize_with_gemma(input_text):

    print("\n[/] request received. generating summary...")

    summary = "[default summary] empty"
    try:
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{
            "role": "user",
            "content": "Generate a summary of the following text:\n" + input_text
            }],
        )

        summary = response['message']['content'].strip()
        processing_time = time.time() - start_time
        print(f"[/] processed in {processing_time} secs")

    except Exception as e:
        print(e)
        print("[/] returning default response")

    return summary

demo = gr.Interface(
        fn=summarize_with_gemma,
        inputs=gr.TextArea(label="Enter a paragraph to create a summary"),
        outputs=["text"],
    )
demo.launch(share=True)
