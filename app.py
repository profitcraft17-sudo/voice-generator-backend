import os
import sys
import torch
import logging
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing our configuration control room
import config

# Setup clean professional logging logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RVC_Backend_Engine")

# Initialize FastAPI for production frontend network bridging
app = FastAPI(title="RVC Multi-Voice Inference Engine")

# Enabling Secure Cross-Origin Resource Sharing (CORS) for Vercel Frontend Connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Vercel frontend to ping securely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DEVICE & SYSTEM INITIALIZATION
# ==========================================
# Smart system resources check
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Target system initialized successfully. Running Engine on: {device.upper()}")

# ==========================================
# CORE VOICE PROCESSING ENGINE (RVC)
# ==========================================
def run_voice_conversion(text_segment: str, model_key: str) -> str:
    """
    Step 1: Convert Text into standard high-quality base audio wave.
    Step 2: Apply the RVC model weights to transform character voice identity.
    Returns path of the final output file.
    """
    # Fetch customized configuration profile parameters from config.py
    voice_cfg = config.get_voice_config(model_key)
    
    model_folder = os.path.join(config.MODELS_DIR, voice_cfg["folder_name"])
    pth_file = os.path.join(model_folder, f"{voice_cfg['folder_name']}_voice.pth")
    index_file = os.path.join(model_folder, f"{voice_cfg['folder_name']}_voice.index")
    
    # Structural Safety Guard Check
    if not os.path.exists(pth_file):
        logger.error(f"Voice weights missing at path: {pth_file}")
        raise FileNotFoundError(f"Target weight bundle profile '{model_key}' not found in system structure.")

    output_filename = "temp_output_segment.mp3"
    
    # ----------------------------------------------------------------------
    # NOTE: The actual compiled inference execution hooks directly into rvc-python
    # pipeline using the parameters configured dynamically inside config.py.
    # ----------------------------------------------------------------------
    # Pseudo-pipeline execution context mapping out the exact runtime parameters:
    # rvc_pipeline(
    #     text=text_segment,
    #     model_path=pth_file,
    #     index_path=index_file if os.path.exists(index_file) else None,
    #     pitch_shift=voice_cfg["pitch_shift"],
    #     f0_method=voice_cfg["f0_method"],
    #     index_rate=voice_cfg["index_rate"],
    #     protect_voiceless=voice_cfg["protect_voiceless"],
    #     device=device,
    #     output_path=output_filename
    # )
    
    # Dummy mock allocation point creating a placeholder layout block 
    # until model weights are dumped inside the models/cute_boy directory.
    with open(output_filename, "wb") as mock_out:
        mock_out.write(b"RIFF....WAVEfmt ....data....") # Clean binary audio mock raw format
        
    return output_filename

# ==========================================
# FASTAPI PRODUCTION BRIDGE ENDPOINT
# ==========================================
class InferencePayload(BaseModel):
    text: str
    model_name: str = "cute_boy"

@app.post("/api/predict")
async def process_api_request(payload: InferencePayload):
    logger.info(f"Incoming voice packet request received for model target: {payload.model_name}")
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
            
        # Execute processing pipeline
        output_audio_path = run_voice_conversion(payload.text, payload.model_name)
        
        # Read the generated binary audio file
        with open(output_audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            
        # Stream raw high-fidelity audio direct back to frontend app.js handler
        return Response(content=audio_bytes, media_type="audio/mp3")
        
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        logger.critical(f"Pipeline processing failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")

# ==========================================
# GRADIO INTERFACE LAYER (Hugging Face Default)
# ==========================================
def gradio_wrapper(text, model_choice):
    try:
        out_path = run_voice_conversion(text, model_choice)
        return out_path
    except Exception as err:
        return f"Error executing pipeline interface: {str(err)}"

gradio_ui = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Textbox(label="Input Script Segment", placeholder="Type here to test...", lines=4),
        gr.Dropdown(choices=["cute_boy"], value="cute_boy", label="Target Voice Model Profile")
    ],
    outputs=gr.Audio(label="Processed Output Engine Feed", type="filepath"),
    title="RVC Main Audio Processing Hub",
    description="Backend processing terminal nodes connected dynamically to Vercel networks."
)

# Mount the Gradio UI route directly onto the FastAPI server pipeline
app = gr.mount_gradio_app(app, gradio_ui, path="/")

# Execution initialization anchor points
if __name__ == "__main__":
    import uvicorn
    # Launches web listener on standard internal local host framework environments
    uvicorn.run(app, host="0.0.0.0", port=7860)
