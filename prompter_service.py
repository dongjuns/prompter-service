import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
app = FastAPI(title="Prompter Service")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# This is the "secret sauce" of our service
PROMPTER_SYSTEM_PROMPT = """
You are 'Prompter', an expert prompt engineer. Your job is to take a user's (Korean) query and perform a 2-step process:

1.  **Enhance & Translate (Step 1):** First, enhance the user's simple query into a detailed, high-quality, and actionable *English* prompt for a main LLM. This prompt should be clear, specific, and anticipate the user's full needs.
2.  **Back-Translate (Step 2):** Second, you must take the *exact English prompt you just generated* and translate it back into *Korean* for user verification.

You **must** return ONLY a JSON object with this structure:
{
  "enhanced_eng_prompt": "The detailed English prompt you created in Step 1",
  "back_translation_kor": "The Korean back-translation from Step 2"
}
"""

# --- API Models ---
class PrompterRequest(BaseModel):
    user_query: str  # This is KOR 1.0

class PrompterResponse(BaseModel):
    enhanced_eng_prompt: str  # This is ENG 2.0
    back_translation_kor: str # This is KOR 2.0

# --- API Endpoint ---
@app.post("/refine", response_model=PrompterResponse)
async def refine_prompt(request: PrompterRequest):
    """
    Takes a simple user query (KOR 1.0) and returns an
    enhanced English prompt (ENG 2.0) and its Korean
    back-translation (KOR 2.0) for verification.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            # Enforce JSON output for reliable parsing
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": PROMPTER_SYSTEM_PROMPT},
                {"role": "user", "content": request.user_query}
            ],
            temperature=0.3 # Low temp for deterministic prompt generation
        )
        
        # The response IS the JSON string, which FastAPI/Pydantic
        # will automatically parse into our PrompterResponse model.
        # We just need to load the string content first.
        import json
        response_data = json.loads(completion.choices[0].message.content)

        return PrompterResponse(
            enhanced_eng_prompt=response_data.get("enhanced_eng_prompt"),
            back_translation_kor=response_data.get("back_translation_kor")
        )

    except Exception as e:
        # In a real startup, we'd have robust error logging here
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn prompter_service:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
