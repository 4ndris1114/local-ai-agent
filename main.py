# Imports
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
import pandas as pd

# Model setup
model = OllamaLLM(model="llama3.2")

# Global state
uploaded_df = None
uploaded_data = []
user_prompt_template = ChatPromptTemplate.from_template(
    "Use the following structured data to answer the question.\n\nData:\n{data}\n\nQuestion:\n{question}"
)

# Upload CSV
def upload_csv(file):
    global uploaded_df, uploaded_data
    if file is None:
        return "Please upload a valid CSV file."
    try:
        uploaded_df = pd.read_csv(file.name)
        uploaded_data = uploaded_df.astype(str).apply(
            lambda row: ", ".join(f"{col}: {val}" for col, val in row.items()), axis=1
        ).tolist()
        return f"CSV uploaded with {len(uploaded_data)} rows!"
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

# Set user-defined prompt
def set_prompt(prompt_text):
    global user_prompt_template
    if "{data}" not in prompt_text or "{question}" not in prompt_text:
        return "Prompt must include both '{data}' and '{question}'."
    try:
        user_prompt_template = ChatPromptTemplate.from_template(prompt_text)
        return "Prompt template updated successfully!"
    except Exception as e:
        return f"Error in prompt template: {str(e)}"

# Answer question using uploaded data and prompt
def answer_question_with_uploaded_data(question):
    if not uploaded_data:
        return "Please upload a CSV file first."
    if user_prompt_template is None:
        return "Prompt template is missing or invalid."
    
    data_text = "\n".join(uploaded_data[:20])  # Limit for performance
    chain = user_prompt_template | model
    result = chain.invoke({"data": data_text, "question": question})
    return result

# Gradio UI
with gr.Blocks(theme="soft") as app:
    gr.Markdown("## Custom CSV AI Assistant")
    gr.Markdown("""
    1. Upload your CSV file  
    2. Review or edit the prompt using `{data}` and `{question}`  
    3. Ask your question!
    """)

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_output = gr.Textbox(label="Upload Status", interactive=False)
    upload_button = gr.Button("Upload CSV")

    default_prompt = (
        "Use the following structured data to answer the question.\n\n"
        "Data:\n{data}\n\n"
        "Question:\n{question}"
    )
    prompt_input = gr.Textbox(
        label="Custom Prompt", lines=6, value=default_prompt
    )
    prompt_status = gr.Textbox(label="Prompt Status", value="Prompt template is ready!", interactive=False)

    question_input = gr.Textbox(label="Your Question", lines=2)
    answer_output = gr.Textbox(label="AI Answer")

    upload_button.click(upload_csv, inputs=file_input, outputs=upload_output)
    prompt_input.change(set_prompt, inputs=prompt_input, outputs=prompt_status)

    ask_button = gr.Button("Ask AI")
    ask_button.click(answer_question_with_uploaded_data, inputs=question_input, outputs=answer_output)

app.launch(server_name="0.0.0.0", server_port=7860)