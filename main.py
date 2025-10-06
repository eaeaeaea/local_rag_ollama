#!/usr/bin/env python3
# Entrypoint so you can: python main.py  (or) uvicorn server:app

import os
import uvicorn

if __name__ == "__main__":
    print(
        "[paths]",
        "BASE_DIR=", os.path.dirname(os.path.abspath(__file__)),
    )
    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
