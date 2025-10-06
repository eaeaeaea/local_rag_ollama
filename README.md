# Local RAG App — **User Guide**

Ask questions over **your own files** and see two answers side-by-side:
- **RAG** (grounded in your uploaded docs)
- **Bare LLM** (the model’s best guess without your docs)

_No coding needed._

---


https://github.com/user-attachments/assets/fdf67be9-00c7-4bc7-b08d-1b86e703c384


## ✅ Requirements (one-time)

- **Python 3.11+**
- **Ollama** installed & running: <https://ollama.com>  
  Pull these models once:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

---

## 🛠 Install (one-time)

From the project folder:
```bash
pip install fastapi uvicorn "pydantic<3" faiss-cpu numpy requests pypdf
  Troubleshooting:
    pip install python-multipart
    pip uninstall urllib3
    pip install 'urllib3<2.0'
```

---

## 🚀 Start

```bash
python main.py
```
Open: **http://localhost:8000**

---

## 🧭 Use the App (3 steps)

1. **Upload**  
   Click **Upload** and choose any mix of **.pdf / .txt / .md / .csv**.  
   *Tip:* CSVs are indexed **row-by-row**, so you can ask row-level questions.

2. **Build Index**  
   Click **Build** (defaults are fine). This prepares your files for search.

3. **Ask**  
   Type your question and press **Ask**.  
   - Left: **RAG** answer citing retrieved chunks  
   - Right: **Bare LLM** answer (no documents used)  
   Expand **Retrieved Chunks** to see which files/rows were used.

---

## 📝 Tips

- Your data stays **local** on your machine.
- If answers feel off, increase **Top-K** (e.g., 8–10) or rephrase the question.
- Scanned PDFs need OCR first (export a text-based PDF or paste into `.txt`).

---

## 🆘 Troubleshooting

- **“No index yet”** → You uploaded files but didn’t click **Build**.  
- **Model/embedding error** → Make sure Ollama is running and you pulled the two models above.  
- **Upload fails** → Only `.pdf/.txt/.md/.csv` are accepted. Try a small `.txt` to test.

---

## 🔒 Privacy

Everything runs locally. Your files are stored in `data/`; the search index is in `artifacts/`.  
Use the **Clear** button to delete uploads at any time.
