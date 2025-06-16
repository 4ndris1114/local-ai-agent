# imports
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import gradio as gr

# Load the local Ollama model
model = OllamaLLM(
    model="llama3.2",
    
)

# Set up the prompt
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Create the prompt template and chain
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to answer questions
def answer_question(question):
    if question.lower() == "q":
        return "Goodbye!"
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    return result

# Launch Gradio app
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the pizza restaurant..."),
    outputs="text",
    title="Pizza Restaurant AI Assistant",
    description="Ask questions about the pizza restaurant based on real customer reviews."
)

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Your Question", lines=2, placeholder="E.g., How is the pizza crust?"),
    outputs=gr.Textbox(label="Answer"),
    title="Pizza Restaurant AI Assistant",
    description="Ask anything about the pizza restaurant. Answers are based on real customer reviews!",
    theme="soft"
)

# localhost:7860
iface.launch(server_name="0.0.0.0", server_port=7860)