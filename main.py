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
user_prompt_template = None

# Upload CSV
def upload_csv(file):
    global uploaded_df
    if file is None:
        return gr.update(choices=[], visible=False), "Please upload a valid CSV file."
    try:
        uploaded_df = pd.read_csv(file.name)
        columns = uploaded_df.columns.tolist()
        return gr.update(choices=columns, visible=True), "CSV uploaded! Select the column with text."
    except Exception as e:
        return gr.update(choices=[], visible=False), f"Error reading CSV: {str(e)}"

# Select column
def set_data_column(column_name):
    global uploaded_data, uploaded_df
    if column_name not in uploaded_df.columns:
        return "Column not found."
    uploaded_data = uploaded_df[column_name].dropna().astype(str).tolist()
    return f"Column '{column_name}' selected with {len(uploaded_data)} entries."

# Set user-defined prompt
def set_prompt(prompt_text):
    global user_prompt_template
    if "{data}" not in prompt_text or "{question}" not in prompt_text:
        return "Prompt must include both '{data}' and '{question}'."
    try:
        user_prompt_template = ChatPromptTemplate.from_template(prompt_text)
        return "Prompt template set successfully!"
    except Exception as e:
        return f"Error in prompt template: {str(e)}"

# Answer question using uploaded data and prompt
def answer_question_with_uploaded_data(question):
    if not uploaded_data:
        return "Please upload and select a column first."
    if user_prompt_template is None:
        return "Please set a valid prompt template first."
    
    data_text = "\n".join(uploaded_data[:20])  # limit for performance
    chain = user_prompt_template | model
    result = chain.invoke({"data": data_text, "question": question})
    return result

# Gradio UI
with gr.Blocks(theme="soft") as app:
    gr.Markdown("## Custom CSV AI Assistant")
    gr.Markdown("""
    1. Upload your CSV file  
    2. Choose the column with text data  
    3. Write a custom prompt using `{data}` and `{question}`  
    4. Ask your question!
    """)

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_output = gr.Textbox(label="Upload Status", interactive=False)
    upload_button = gr.Button("Upload CSV")

    column_dropdown = gr.Dropdown(label="Select Text Column", choices=[], visible=False)
    column_status = gr.Textbox(label="Column Status", interactive=False)

    prompt_input = gr.Textbox(label="Custom Prompt", lines=5, placeholder="E.g., Use the following data to answer the question:\n\nData:{data}\n\nQuestion: {question}")
    prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

    question_input = gr.Textbox(label="Your Question", lines=2)
    answer_output = gr.Textbox(label="AI Answer")

    upload_button.click(upload_csv, inputs=file_input, outputs=[column_dropdown, upload_output])

    column_dropdown.change(set_data_column, inputs=column_dropdown, outputs=column_status)

    prompt_input.change(set_prompt, inputs=prompt_input, outputs=prompt_status)

    ask_button = gr.Button("Ask AI")
    ask_button.click(answer_question_with_uploaded_data, inputs=question_input, outputs=answer_output)

app.launch(server_name="0.0.0.0", server_port=7860)