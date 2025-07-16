import os
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
import gradio as gr

# Set your Hugging Face API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_api_key"

# Function to read PDF resume
def read_resume(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    pages = loader.load()
    resume_text = "\n".join([page.page_content for page in pages])
    return resume_text

# Function to generate cover letter
def generate_cover_letter(resume_file, job_description):
    resume_text = read_resume(resume_file)

    # Load LLM (Zephyr 7B works well for this)
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.7, "max_length": 1000}
    )

    # Define the prompt template
    template = """
    You are a professional career advisor and expert writer.
    Based on the following resume and job description, write a highly tailored and professional cover letter.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Please make the cover letter formal, to-the-point, and personalized for this job role.
    """

    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the cover letter
    cover_letter = chain.run({"resume": resume_text, "job_description": job_description})
    return cover_letter

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# AI Cover Letter Generator 💼")

    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF)")
        jd_input = gr.Textbox(label="Paste Job Description (JD)", lines=6)

    output = gr.Textbox(label="Generated Cover Letter", lines=12)

    submit_btn = gr.Button("Generate Cover Letter")
    submit_btn.click(generate_cover_letter, inputs=[resume_input, jd_input], outputs=output)

app.launch()
