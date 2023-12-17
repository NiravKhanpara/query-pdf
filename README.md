# query-pdf

## Overview

This tool is designed to simplify PDF document querying using LangChain and the GooglePaLM API. It converts PDF text into FAISS vectors and employs Retrieval-Augmented Generation for efficient user queries.

## How it Works

1. **Upload PDF:** Use the file input menu to upload your PDF document.
2. **Text Extraction:** Extract text from the uploaded PDF.
3. **Vector Database:** Store PDF text in a vector database using embeddings.
4. **Query Processing:** Find similar text in the vector database based on user queries.
5. **Output Generation:** Generate and display the final output using the LLM model.

## Contributing

We welcome contributions! Feel free to fork the repository, open issues, and submit pull requests.