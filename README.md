The Document Summarization App allows users to upload PDF documents, which are then processed and summarized using the **LaMini-Flan-T5-248M** model from Hugging Face Transformers. This app is built using Streamlit, an open-source Python framework, providing an easy-to-use interface for users. It combines several powerful tools to split, process, and summarize documents effectively.

Features
**PDF Upload**: Allows users to upload PDF files for summarization.
**Summarization**: The uploaded document is processed using a pre-trained transformer model to generate a concise summary.
**PDF Display**: The original PDF is displayed alongside the summary for easy comparison.
**Easy-to-Use UI**: Streamlit interface with minimal effort required for interaction.
Technologies
This app leverages several key technologies:

Streamlit: A Python framework for creating interactive web apps with minimal coding effort.
Hugging Face Transformers: A library used to load the LaMini-Flan-T5-248M model for text summarization.
Torch: A deep learning library required for model execution.
LangChain: A library for document splitting to efficiently handle larger PDFs.
PyPDF2: Used for loading and processing PDF documents.
