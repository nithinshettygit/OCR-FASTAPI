from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from typing import List
from backend.amount_extraction_engine import extract_max_amount
logger = logging.getLogger("uvicorn.error")

# 1. Initialize the app instance ONCE
app = FastAPI()

# 2. Add CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://43.204.98.239"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API!", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/extract-amount")
async def extract_amount(files: List[UploadFile] = File(...)):
    logger.info("OCR request started: %d file(s)", len(files))
    results = []
    errors = []

    for file in files:
        temp_path = None
        try:
            # Secure the correct file extension
            suffix = os.path.splitext(file.filename)[1]

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            # Run OCR engine
            result = extract_max_amount(
                temp_path,
                return_debug=False
            )
            logger.info("OCR completed: %s", file.filename)
            
            # Append success payload
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })

        except Exception as e:
            logger.exception("OCR failed: %s", file.filename)
            # Gather individual file errors without crashing the entire batch request
            errors.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
            
        finally:
            # Ensure the cleanup runs even if the OCR engine fails mid-execution
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    logger.info(
        "OCR request finished: %d succeeded, %d failed",
        len(results),
        len(errors),
    )

    # Return a structured batch summary payload
    return {
        "processed_count": len(results),
        "failed_count": len(errors),
        "data": results + errors
    }
